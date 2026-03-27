"""Tests for production trace dataset loading."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from inferscope.benchmarks.datasets import (
    DatasetLoadOptions,
    _convert_sharegpt_conversation,
    _estimate_tokens,
    _parse_hf_reference,
    _reservoir_sample,
    load_trace_dataset,
)


# ---------------------------------------------------------------------------
# Token estimation
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic() -> None:
    assert _estimate_tokens("hello world here!") >= 4


def test_estimate_tokens_empty() -> None:
    assert _estimate_tokens("") == 1


# ---------------------------------------------------------------------------
# HuggingFace reference parsing
# ---------------------------------------------------------------------------


def test_parse_hf_reference_simple() -> None:
    repo, config, split = _parse_hf_reference("lmsys/lmsys-chat-1m")
    assert repo == "lmsys/lmsys-chat-1m"
    assert config is None
    assert split is None


def test_parse_hf_reference_with_split() -> None:
    repo, config, split = _parse_hf_reference("lmsys/lmsys-chat-1m#train")
    assert repo == "lmsys/lmsys-chat-1m"
    assert split == "train"


def test_parse_hf_reference_with_config_and_split() -> None:
    repo, config, split = _parse_hf_reference("OpenAssistant/oasst1@default#validation")
    assert repo == "OpenAssistant/oasst1"
    assert config == "default"
    assert split == "validation"


def test_parse_hf_reference_with_config_only() -> None:
    repo, config, split = _parse_hf_reference("org/dataset@my-config")
    assert repo == "org/dataset"
    assert config == "my-config"
    assert split is None


# ---------------------------------------------------------------------------
# ShareGPT conversion
# ---------------------------------------------------------------------------


def _default_options() -> DatasetLoadOptions:
    return DatasetLoadOptions(sample_size=10, seed=42)


def test_convert_sharegpt_basic() -> None:
    row = {
        "id": "abc123",
        "conversations": [
            {"from": "human", "value": "What is Python?"},
            {"from": "gpt", "value": "Python is a programming language."},
        ],
    }
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is not None
    assert req.name == "sharegpt-000000"
    assert len(req.messages) == 1  # assistant trimmed
    assert req.messages[0].role == "user"
    assert req.messages[0].content == "What is Python?"
    assert req.max_tokens > 0
    assert req.metadata["dataset_source"] == "sharegpt"


def test_convert_sharegpt_multi_turn() -> None:
    row = {
        "id": "multi",
        "conversations": [
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
            {"from": "human", "value": "Tell me about AI"},
            {"from": "gpt", "value": "AI is artificial intelligence."},
        ],
    }
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is not None
    # Last assistant trimmed, rest kept
    assert len(req.messages) == 3
    assert req.messages[0].role == "user"
    assert req.messages[1].role == "assistant"
    assert req.messages[2].role == "user"


def test_convert_sharegpt_empty_conversations() -> None:
    row = {"id": "empty", "conversations": []}
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is None


def test_convert_sharegpt_no_user_message() -> None:
    row = {
        "id": "no-user",
        "conversations": [
            {"from": "gpt", "value": "I have no question to answer."},
        ],
    }
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is None


def test_convert_sharegpt_unknown_role_skipped() -> None:
    row = {
        "id": "weird",
        "conversations": [
            {"from": "alien", "value": "Greetings"},
        ],
    }
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is None


def test_convert_sharegpt_system_message_preserved() -> None:
    row = {
        "id": "sys",
        "conversations": [
            {"from": "system", "value": "You are helpful."},
            {"from": "human", "value": "Hi"},
            {"from": "gpt", "value": "Hello!"},
        ],
    }
    req = _convert_sharegpt_conversation(row, 0, _default_options(), "test")
    assert req is not None
    assert req.messages[0].role == "system"
    assert req.messages[1].role == "user"


def test_convert_sharegpt_max_tokens_clamped() -> None:
    # Very long assistant response
    long_response = "x" * 50000  # ~12500 tokens
    row = {
        "id": "long",
        "conversations": [
            {"from": "human", "value": "Write a lot"},
            {"from": "gpt", "value": long_response},
        ],
    }
    opts = DatasetLoadOptions(max_tokens_cap=2048)
    req = _convert_sharegpt_conversation(row, 0, opts, "test")
    assert req is not None
    assert req.max_tokens == 2048  # clamped


# ---------------------------------------------------------------------------
# Reservoir sampling
# ---------------------------------------------------------------------------


def test_reservoir_sample_basic() -> None:
    items = [{"conversations": [{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}]} for i in range(100)]
    opts = _default_options()
    results = _reservoir_sample(
        items,
        convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, opts, "test"),
        sample_size=10,
        seed=42,
    )
    assert len(results) == 10


def test_reservoir_sample_deterministic() -> None:
    items = [{"conversations": [{"from": "human", "value": f"Q{i}"}, {"from": "gpt", "value": f"A{i}"}]} for i in range(50)]
    opts = _default_options()
    r1 = _reservoir_sample(items, convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, opts, "test"), sample_size=5, seed=42)
    r2 = _reservoir_sample(items, convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, opts, "test"), sample_size=5, seed=42)
    assert [r.name for r in r1] == [r.name for r in r2]


def test_reservoir_sample_fewer_than_sample_size() -> None:
    items = [{"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]} for _ in range(3)]
    opts = _default_options()
    results = _reservoir_sample(
        items,
        convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, opts, "test"),
        sample_size=100,
        seed=42,
    )
    assert len(results) == 3


def test_reservoir_sample_skips_invalid() -> None:
    items = [
        {"conversations": []},  # invalid
        {"conversations": [{"from": "human", "value": "Q"}, {"from": "gpt", "value": "A"}]},  # valid
        {"bad": True},  # invalid
    ]
    opts = _default_options()
    results = _reservoir_sample(
        items,
        convert_fn=lambda row, idx: _convert_sharegpt_conversation(row, idx, opts, "test"),
        sample_size=10,
        seed=42,
    )
    assert len(results) == 1


# ---------------------------------------------------------------------------
# Local file loading: ShareGPT JSON
# ---------------------------------------------------------------------------


def test_load_sharegpt_json_file() -> None:
    data = [
        {
            "id": f"conv-{i}",
            "conversations": [
                {"from": "human", "value": f"Question {i} about programming"},
                {"from": "gpt", "value": f"Here is a detailed answer about topic {i}."},
            ],
        }
        for i in range(20)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    try:
        pack = load_trace_dataset(path, options=DatasetLoadOptions(sample_size=5, seed=42))
        assert pack.name == "sharegpt"
        assert len(pack.requests) == 5
        assert pack.workload_class == "chat"
        assert "hydration:hydrated" in pack.tags
        assert "dataset:sharegpt" in pack.tags
        assert pack.requests[0].metadata["dataset_source"] == "sharegpt"
    finally:
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Local file loading: JSONL requests
# ---------------------------------------------------------------------------


def test_load_jsonl_file() -> None:
    lines = [
        json.dumps({
            "messages": [
                {"role": "user", "content": f"Hello from request {i}"},
            ],
            "max_tokens": 128,
        })
        for i in range(10)
    ]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("\n".join(lines) + "\n")
        path = f.name

    try:
        pack = load_trace_dataset(path, options=DatasetLoadOptions(sample_size=5, seed=42))
        assert len(pack.requests) == 5
        assert pack.workload_class == "chat"
        assert "dataset:jsonl" in pack.tags
    finally:
        Path(path).unlink()


def test_load_jsonl_with_tools() -> None:
    data = {
        "messages": [{"role": "user", "content": "Call the weather API"}],
        "max_tokens": 64,
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(data) + "\n")
        path = f.name

    try:
        pack = load_trace_dataset(path)
        assert pack.workload_class == "agent"
        assert pack.requests[0].tools is not None
    finally:
        Path(path).unlink()


def test_load_jsonl_invalid_json_fails() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"messages": [{"role": "user", "content": "ok"}]}\n')
        f.write("not valid json\n")
        path = f.name

    try:
        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_trace_dataset(path)
    finally:
        Path(path).unlink()


def test_load_jsonl_missing_messages_fails() -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write('{"prompt": "hello"}\n')
        path = f.name

    try:
        with pytest.raises(ValueError, match="messages"):
            load_trace_dataset(path)
    finally:
        Path(path).unlink()


# ---------------------------------------------------------------------------
# Reference resolution
# ---------------------------------------------------------------------------


def test_unknown_reference_without_slash_fails() -> None:
    with pytest.raises(ValueError, match="Unknown dataset reference"):
        load_trace_dataset("not-a-valid-ref")


def test_unsupported_file_extension_fails() -> None:
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        f.write(b"a,b,c\n1,2,3\n")
        path = f.name

    try:
        with pytest.raises(ValueError, match="Unsupported local file format"):
            load_trace_dataset(path)
    finally:
        Path(path).unlink()


# ---------------------------------------------------------------------------
# DatasetLoadOptions validation
# ---------------------------------------------------------------------------


def test_dataset_load_options_defaults() -> None:
    opts = DatasetLoadOptions()
    assert opts.sample_size == 128
    assert opts.seed == 42
    assert opts.split == "train"
    assert opts.stream is True
    assert opts.endpoint_path == "/v1/chat/completions"
    assert opts.default_max_tokens == 512
    assert opts.max_tokens_cap == 4096
