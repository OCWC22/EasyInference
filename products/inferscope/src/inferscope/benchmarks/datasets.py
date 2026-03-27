"""Production trace dataset loading for real-world benchmark workloads.

Supports:
- ShareGPT V3 format (JSON array of conversations) — the universal LLM serving benchmark standard
- JSONL request format (one OpenAI-compatible request per line)
- HuggingFace datasets (any conversational dataset, loaded via the `datasets` library)

These loaders convert real production traces into WorkloadPack objects that the benchmark
runtime can replay against a live inference endpoint.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from inferscope.benchmarks.models import ChatMessage, WorkloadPack, WorkloadRequest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SHAREGPT_HF_REPO = "anon8231489123/ShareGPT_Vicuna_unfiltered"
_SHAREGPT_HF_FILE = "ShareGPT_V3_unfiltered_cleaned_split.json"

# Role mapping from ShareGPT to OpenAI format
_SHAREGPT_ROLE_MAP: dict[str, str] = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
}

# Built-in dataset aliases
_DATASET_ALIASES: dict[str, dict[str, Any]] = {
    "sharegpt": {
        "source": "huggingface",
        "repo": _SHAREGPT_HF_REPO,
        "file": _SHAREGPT_HF_FILE,
        "adapter": "sharegpt",
        "split": "train",
        "description": "ShareGPT V3 — 53K real ChatGPT conversations, universal serving benchmark standard",
    },
}

# HF row field detection for generic adapter
_PROMPT_FIELDS = ("prompt", "instruction", "question", "input", "query", "text")
_RESPONSE_FIELDS = ("response", "output", "answer", "completion", "chosen")


class DatasetLoadOptions(BaseModel):
    """Configuration for loading a trace dataset into a WorkloadPack."""

    model_config = ConfigDict(extra="forbid")

    sample_size: int = Field(default=128, ge=1, description="Number of conversations to sample")
    seed: int = Field(default=42, description="Random seed for deterministic sampling")
    split: str = Field(default="train", description="HuggingFace dataset split")
    config_name: str | None = Field(default=None, description="HuggingFace dataset config")
    stream: bool = Field(default=True, description="Whether to stream SSE responses during replay")
    endpoint_path: str = Field(default="/v1/chat/completions", description="API endpoint path")
    concurrency: int = Field(default=1, ge=1, description="Concurrent requests during replay")
    default_max_tokens: int = Field(default=512, ge=1, description="Default max_tokens when not inferrable")
    max_tokens_cap: int = Field(default=4096, ge=32, description="Upper bound clamp for max_tokens")


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return max(1, math.ceil(len(text) / 4))


# ---------------------------------------------------------------------------
# ShareGPT conversion
# ---------------------------------------------------------------------------


def _convert_sharegpt_conversation(
    row: dict[str, Any],
    row_index: int,
    options: DatasetLoadOptions,
    reference: str,
) -> WorkloadRequest | None:
    """Convert a single ShareGPT conversation to a WorkloadRequest, or None if invalid."""
    conversations = row.get("conversations") or row.get("conversation")
    if not conversations or not isinstance(conversations, list):
        return None

    # Convert roles and filter
    messages: list[ChatMessage] = []
    for turn in conversations:
        # Handle both {from, value} and {role, content} formats
        role_raw = turn.get("from") or turn.get("role", "")
        content = turn.get("value") or turn.get("content", "")
        role = _SHAREGPT_ROLE_MAP.get(role_raw, "")
        if not role or not content or not content.strip():
            continue
        messages.append(ChatMessage(role=role, content=content))

    if not messages:
        return None

    # Must have at least one user message
    if not any(m.role == "user" for m in messages):
        return None

    # If final message is assistant, trim it and use for max_tokens estimation
    max_tokens = options.default_max_tokens
    if messages[-1].role == "assistant":
        final_assistant = messages.pop()
        estimated = _estimate_tokens(final_assistant.content)
        max_tokens = max(32, min(estimated, options.max_tokens_cap))

    # After trimming, must still have messages with at least one user turn
    if not messages or not any(m.role == "user" for m in messages):
        return None

    # Estimate prompt tokens for metadata
    prompt_tokens = sum(_estimate_tokens(m.content) + _estimate_tokens(m.role) for m in messages)

    row_id = row.get("id", f"row-{row_index}")
    return WorkloadRequest(
        name=f"sharegpt-{row_index:06d}",
        session_id=str(row_id) if row_id else None,
        messages=messages,
        max_tokens=max_tokens,
        metadata={
            "dataset_source": "sharegpt",
            "dataset_reference": reference,
            "dataset_record_id": str(row_id),
            "target_context_tokens": prompt_tokens,
        },
    )


def _load_sharegpt_from_json(path: Path, options: DatasetLoadOptions, reference: str) -> WorkloadPack:
    """Load a local ShareGPT JSON file."""
    with open(path) as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"ShareGPT JSON must be a top-level array, got {type(data).__name__}")

    requests = _reservoir_sample(
        (row for row in data),
        convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, options, reference),
        sample_size=options.sample_size,
        seed=options.seed,
    )

    if not requests:
        raise ValueError(f"ShareGPT file produced zero valid benchmark requests: {path}")

    return _build_dataset_pack(
        name="sharegpt",
        description=f"ShareGPT V3 production conversations ({len(requests)} sampled from {path.name})",
        workload_class="chat",
        requests=requests,
        options=options,
        tags=["dataset", "dataset:sharegpt", "source:production", "hydration:hydrated"],
    )


# ---------------------------------------------------------------------------
# JSONL request loading
# ---------------------------------------------------------------------------


def _load_jsonl_requests(path: Path, options: DatasetLoadOptions) -> WorkloadPack:
    """Load a JSONL file where each line is an OpenAI-compatible request."""
    requests: list[WorkloadRequest] = []
    has_tools = False

    with open(path) as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc

            if not isinstance(data, dict):
                raise ValueError(f"Line {line_num} of {path}: expected JSON object, got {type(data).__name__}")

            messages_raw = data.get("messages")
            if not messages_raw or not isinstance(messages_raw, list):
                raise ValueError(f"Line {line_num} of {path}: 'messages' field is required and must be a list")

            # Convert messages
            messages = [
                ChatMessage(
                    role=m.get("role", "user"),
                    content=m.get("content", ""),
                    name=m.get("name"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in messages_raw
                if isinstance(m, dict)
            ]

            if not messages:
                raise ValueError(f"Line {line_num} of {path}: no valid messages found")

            name = data.get("name", f"request-{line_num:06d}")
            max_tokens = data.get("max_tokens", options.default_max_tokens)
            max_tokens = max(32, min(int(max_tokens), options.max_tokens_cap))

            # Detect tools
            tools = data.get("tools")
            if tools:
                has_tools = True

            prompt_tokens = sum(_estimate_tokens(m.content or "") + _estimate_tokens(m.role) for m in messages)

            metadata = data.get("metadata", {})
            metadata.update({
                "dataset_source": "jsonl",
                "dataset_reference": str(path),
                "dataset_line": line_num,
                "target_context_tokens": metadata.get("target_context_tokens", prompt_tokens),
            })

            # Build request kwargs, omitting None values for fields with non-None defaults
            req_kwargs: dict[str, Any] = {
                "name": name,
                "messages": messages,
                "max_tokens": max_tokens,
                "metadata": metadata,
            }
            if data.get("session_id") is not None:
                req_kwargs["session_id"] = data["session_id"]
            if data.get("temperature") is not None:
                req_kwargs["temperature"] = data["temperature"]
            if tools:
                req_kwargs["tools"] = tools
            if data.get("tool_choice") is not None:
                req_kwargs["tool_choice"] = data["tool_choice"]
            if data.get("headers"):
                req_kwargs["headers"] = data["headers"]
            if data.get("extra_body"):
                req_kwargs["extra_body"] = data["extra_body"]
            if data.get("think_time_ms") is not None:
                req_kwargs["think_time_ms"] = data["think_time_ms"]

            requests.append(WorkloadRequest(**req_kwargs))

    if not requests:
        raise ValueError(f"JSONL file produced zero valid requests: {path}")

    # Sample if needed
    if len(requests) > options.sample_size:
        rng = random.Random(options.seed)
        requests = rng.sample(requests, options.sample_size)

    workload_class = "agent" if has_tools else "chat"
    return _build_dataset_pack(
        name=path.stem,
        description=f"JSONL requests from {path.name} ({len(requests)} requests)",
        workload_class=workload_class,
        requests=requests,
        options=options,
        tags=["dataset", "dataset:jsonl", "source:production", "hydration:hydrated"],
    )


# ---------------------------------------------------------------------------
# HuggingFace dataset loading
# ---------------------------------------------------------------------------


def _parse_hf_reference(reference: str) -> tuple[str, str | None, str | None]:
    """Parse 'repo[@config][#split]' into (repo, config, split)."""
    config = None
    split = None

    if "#" in reference:
        reference, split = reference.rsplit("#", 1)

    if "@" in reference:
        reference, config = reference.rsplit("@", 1)

    return reference, config or None, split or None


def _convert_hf_row(
    row: dict[str, Any],
    row_index: int,
    options: DatasetLoadOptions,
    reference: str,
) -> WorkloadRequest | None:
    """Try to convert a HuggingFace dataset row to a WorkloadRequest using heuristic adapters."""

    # Adapter 1: OpenAI messages format
    if "messages" in row and isinstance(row["messages"], list):
        messages = []
        for m in row["messages"]:
            if isinstance(m, dict) and "role" in m:
                content = m.get("content", "")
                if content and isinstance(content, str):
                    messages.append(ChatMessage(role=m["role"], content=content))
        if messages and any(m.role == "user" for m in messages):
            max_tokens = options.default_max_tokens
            if messages[-1].role == "assistant":
                final = messages.pop()
                max_tokens = max(32, min(_estimate_tokens(final.content), options.max_tokens_cap))
            if messages and any(m.role == "user" for m in messages):
                prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
                return WorkloadRequest(
                    name=f"hf-{row_index:06d}",
                    messages=messages,
                    max_tokens=max_tokens,
                    metadata={
                        "dataset_source": "huggingface",
                        "dataset_reference": reference,
                        "target_context_tokens": prompt_tokens,
                    },
                )

    # Adapter 2: ShareGPT conversation format
    conversations = row.get("conversations") or row.get("conversation")
    if conversations and isinstance(conversations, list):
        req = _convert_sharegpt_conversation(row, row_index, options, reference)
        if req:
            req.name = f"hf-{row_index:06d}"
            return req

    # Adapter 3: Prompt/response format
    for prompt_key in _PROMPT_FIELDS:
        if prompt_key in row and isinstance(row[prompt_key], str) and row[prompt_key].strip():
            prompt_text = row[prompt_key].strip()
            messages = []

            # Optional system message
            if "system" in row and isinstance(row["system"], str) and row["system"].strip():
                messages.append(ChatMessage(role="system", content=row["system"].strip()))

            messages.append(ChatMessage(role="user", content=prompt_text))

            max_tokens = options.default_max_tokens
            for resp_key in _RESPONSE_FIELDS:
                if resp_key in row and isinstance(row[resp_key], str) and row[resp_key].strip():
                    max_tokens = max(32, min(_estimate_tokens(row[resp_key]), options.max_tokens_cap))
                    break

            prompt_tokens = sum(_estimate_tokens(m.content) for m in messages)
            return WorkloadRequest(
                name=f"hf-{row_index:06d}",
                messages=messages,
                max_tokens=max_tokens,
                metadata={
                    "dataset_source": "huggingface",
                    "dataset_reference": reference,
                    "target_context_tokens": prompt_tokens,
                },
            )

    return None


def _load_hf_dataset(
    repo: str,
    options: DatasetLoadOptions,
    reference: str,
    *,
    config_name: str | None = None,
    split: str | None = None,
    data_files: str | None = None,
    adapter: str = "auto",
) -> WorkloadPack:
    """Load a HuggingFace dataset and convert to WorkloadPack."""
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for HuggingFace dataset loading. "
            "Install it with: pip install datasets"
        ) from exc

    effective_split = split or options.split
    effective_config = config_name or options.config_name

    try:
        load_kwargs: dict[str, Any] = {
            "path": repo,
            "name": effective_config,
            "split": effective_split,
            "streaming": True,
            "trust_remote_code": False,
        }
        if data_files:
            load_kwargs["data_files"] = data_files
        ds = load_dataset(**load_kwargs)
    except Exception as exc:
        raise ValueError(
            f"Failed to load HuggingFace dataset '{repo}' "
            f"(config={effective_config}, split={effective_split}): {exc}"
        ) from exc

    # Choose conversion function based on adapter
    if adapter == "sharegpt":
        convert_fn = lambda row, idx: _convert_sharegpt_conversation(row, idx, options, reference)
    else:
        convert_fn = lambda row, idx: _convert_hf_row(row, idx, options, reference)

    requests = _reservoir_sample(
        ds,
        convert_fn=convert_fn,
        sample_size=options.sample_size,
        seed=options.seed,
    )

    if not requests:
        raise ValueError(
            f"HuggingFace dataset '{repo}' produced zero valid benchmark requests. "
            f"The dataset rows may not match any supported conversation format "
            f"(OpenAI messages, ShareGPT conversations, or prompt/response pairs)."
        )

    return _build_dataset_pack(
        name=f"hf-{repo.replace('/', '-')}",
        description=f"HuggingFace dataset {repo} ({len(requests)} sampled, split={effective_split})",
        workload_class="chat",
        requests=requests,
        options=options,
        tags=["dataset", f"dataset:hf:{repo}", "source:production", "hydration:hydrated"],
    )


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------


def _reservoir_sample(
    iterable: Any,
    *,
    convert_fn: Any,
    sample_size: int,
    seed: int,
) -> list[WorkloadRequest]:
    """Deterministic reservoir sampling over an iterable of raw rows.

    Converts each row via convert_fn(row, global_index) -> WorkloadRequest | None.
    Invalid rows (returning None) are skipped.
    """
    rng = random.Random(seed)
    reservoir: list[tuple[int, WorkloadRequest]] = []  # (original_index, request)
    seen = 0

    for global_index, row in enumerate(iterable):
        if not isinstance(row, dict):
            continue
        req = convert_fn(row, global_index)
        if req is None:
            continue

        seen += 1
        if len(reservoir) < sample_size:
            reservoir.append((global_index, req))
        else:
            j = rng.randint(0, seen - 1)
            if j < sample_size:
                reservoir[j] = (global_index, req)

    # Sort by original index for deterministic ordering
    reservoir.sort(key=lambda x: x[0])
    return [req for _, req in reservoir]


# ---------------------------------------------------------------------------
# Pack builder
# ---------------------------------------------------------------------------


def _build_dataset_pack(
    *,
    name: str,
    description: str,
    workload_class: str,
    requests: list[WorkloadRequest],
    options: DatasetLoadOptions,
    tags: list[str],
) -> WorkloadPack:
    """Build a WorkloadPack from loaded dataset requests."""
    return WorkloadPack(
        name=name,
        description=description,
        workload_class=workload_class,
        model="",  # Model is specified at benchmark runtime, not in the workload
        endpoint_path=options.endpoint_path,
        concurrency=options.concurrency,
        stream=options.stream,
        tags=tags,
        requests=requests,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_trace_dataset(
    reference: str,
    *,
    options: DatasetLoadOptions | None = None,
) -> WorkloadPack:
    """Load a production trace dataset into a WorkloadPack for benchmark replay.

    Args:
        reference: One of:
            - "sharegpt" — built-in alias for ShareGPT V3
            - Local file path (.json for ShareGPT format, .jsonl for request format)
            - HuggingFace dataset reference: "repo[@config][#split]"
        options: Loading configuration (sampling, concurrency, etc.)

    Returns:
        A WorkloadPack ready for benchmark replay.

    Raises:
        ValueError: If the dataset cannot be loaded or produces no valid requests.
        ImportError: If the 'datasets' package is required but not installed.

    Examples:
        # Load ShareGPT V3 (downloads from HuggingFace)
        pack = load_trace_dataset("sharegpt")

        # Load a local JSONL file
        pack = load_trace_dataset("my-traces.jsonl", options=DatasetLoadOptions(sample_size=256))

        # Load any HuggingFace conversational dataset
        pack = load_trace_dataset("lmsys/lmsys-chat-1m#train")
    """
    opts = options or DatasetLoadOptions()

    # 1. Check for local file
    local_path = Path(reference)
    if local_path.exists() and local_path.is_file():
        suffix = local_path.suffix.lower()
        if suffix == ".jsonl":
            return _load_jsonl_requests(local_path, opts)
        elif suffix == ".json":
            return _load_sharegpt_from_json(local_path, opts, reference)
        else:
            raise ValueError(
                f"Unsupported local file format '{suffix}'. "
                f"Supported: .json (ShareGPT format), .jsonl (request-per-line format)"
            )

    # 2. Check built-in aliases
    if reference in _DATASET_ALIASES:
        alias = _DATASET_ALIASES[reference]
        return _load_hf_dataset(
            alias["repo"],
            opts,
            reference,
            config_name=alias.get("config"),
            split=alias.get("split"),
            data_files=alias.get("file"),
            adapter=alias.get("adapter", "auto"),
        )

    # 3. Parse as HuggingFace reference
    repo, config, split = _parse_hf_reference(reference)
    if "/" not in repo:
        raise ValueError(
            f"Unknown dataset reference '{reference}'. "
            f"Use a built-in alias ({', '.join(sorted(_DATASET_ALIASES))}), "
            f"a local file path (.json/.jsonl), "
            f"or a HuggingFace dataset (org/name[@config][#split])."
        )

    return _load_hf_dataset(repo, opts, reference, config_name=config, split=split)
