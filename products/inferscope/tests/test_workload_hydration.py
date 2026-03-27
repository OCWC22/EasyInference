"""Tests for workload hydration metadata, token accounting, and hydration modes."""

from __future__ import annotations

import pytest

from inferscope.benchmarks.models import (
    ChatMessage,
    HydrationMode,
    WorkloadPack,
    WorkloadRequest,
    estimate_tokens,
)


# ---------------------------------------------------------------------------
# estimate_tokens
# ---------------------------------------------------------------------------


def test_estimate_tokens_basic_string() -> None:
    # 16 chars → ~4 tokens
    assert estimate_tokens("hello world here") == 4


def test_estimate_tokens_empty_string() -> None:
    # Empty string still returns 1 due to max(1, ceil(0/4)) in the implementation
    assert estimate_tokens("") == 1


def test_estimate_tokens_none() -> None:
    assert estimate_tokens(None) == 0


def test_estimate_tokens_rounds_up() -> None:
    # 5 chars → ceil(5/4) = 2
    assert estimate_tokens("hello") == 2


# ---------------------------------------------------------------------------
# WorkloadRequest.target_context_tokens
# ---------------------------------------------------------------------------


def _make_request(metadata: dict | None = None, content: str = "Hello") -> WorkloadRequest:
    return WorkloadRequest(
        name="test",
        messages=[ChatMessage(role="user", content=content)],
        metadata=metadata or {},
        max_tokens=64,
    )


def test_target_context_tokens_from_metadata() -> None:
    req = _make_request(metadata={"target_context_tokens": 96000})
    assert req.target_context_tokens == 96000


def test_target_context_tokens_legacy_fallback() -> None:
    req = _make_request(metadata={"approx_context_tokens": 8192})
    assert req.target_context_tokens == 8192


def test_target_context_tokens_none_when_absent() -> None:
    req = _make_request(metadata={})
    assert req.target_context_tokens is None


def test_target_context_tokens_prefers_new_key() -> None:
    req = _make_request(metadata={"target_context_tokens": 50000, "approx_context_tokens": 8000})
    assert req.target_context_tokens == 50000


# ---------------------------------------------------------------------------
# WorkloadRequest.actual_context_tokens
# ---------------------------------------------------------------------------


def test_actual_context_tokens_from_content() -> None:
    req = _make_request(content="a" * 400)
    # 400 chars → 100 tokens (content) + estimate_tokens("user") + estimate_tokens("test")
    actual = req.actual_context_tokens
    assert actual > 0
    assert actual >= 100  # At least the content tokens


# ---------------------------------------------------------------------------
# WorkloadPack.hydration_mode
# ---------------------------------------------------------------------------


def _make_pack(tags: list[str] | None = None) -> WorkloadPack:
    return WorkloadPack(
        name="test-pack",
        description="test",
        workload_class="coding",
        model="test-model",
        concurrency=1,
        stream=True,
        tags=tags or [],
        requests=[_make_request()],
    )


def test_hydration_mode_template() -> None:
    pack = _make_pack(tags=["hydration:template", "coding"])
    assert pack.hydration_mode == "template"


def test_hydration_mode_synthetic() -> None:
    pack = _make_pack(tags=["hydration:synthetic"])
    assert pack.hydration_mode == "synthetic"


def test_hydration_mode_hydrated_explicit() -> None:
    pack = _make_pack(tags=["hydration:hydrated"])
    assert pack.hydration_mode == "hydrated"


def test_hydration_mode_default_hydrated() -> None:
    pack = _make_pack(tags=["coding", "long-context"])
    assert pack.hydration_mode == "hydrated"


# ---------------------------------------------------------------------------
# WorkloadPack.max_target_context_tokens / max_actual_context_tokens
# ---------------------------------------------------------------------------


def test_max_target_context_tokens() -> None:
    pack = WorkloadPack(
        name="test-pack",
        description="test",
        workload_class="coding",
        model="test-model",
        concurrency=1,
        stream=True,
        requests=[
            _make_request(metadata={"target_context_tokens": 32000}),
            _make_request(metadata={"target_context_tokens": 96000}),
        ],
    )
    assert pack.max_target_context_tokens() == 96000


def test_max_target_context_tokens_none_when_absent() -> None:
    pack = _make_pack()
    assert pack.max_target_context_tokens() is None


def test_max_actual_context_tokens() -> None:
    pack = WorkloadPack(
        name="test-pack",
        description="test",
        workload_class="coding",
        model="test-model",
        concurrency=1,
        stream=True,
        requests=[
            _make_request(content="short"),
            _make_request(content="a" * 2000),
        ],
    )
    result = pack.max_actual_context_tokens()
    assert result >= 500  # 2000 chars / 4 ≈ 500 tokens
