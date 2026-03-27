"""Tests for MCP-side benchmark resolution helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from inferscope.benchmarks import BenchmarkArtifact, BenchmarkSummary
from inferscope.server_benchmarks import _resolve_benchmark_plan, register_benchmark_tools


def test_resolve_benchmark_plan_supports_procedural_workloads() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "kimi-k2-long-context-coding",
        "http://localhost:8000",
        synthetic_requests=4,
        synthetic_input_tokens=2048,
        synthetic_output_tokens=256,
    )
    assert error is None
    assert workload_reference == "kimi-k2-long-context-coding"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert len(workload_pack.requests) == 4
    assert run_plan.workload_ref == "kimi-k2-long-context-coding"
    assert support.status == "unknown"


def test_resolve_benchmark_plan_rejects_context_file_for_mcp() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "kimi-k2-long-context-coding",
        "http://localhost:8000",
        synthetic_requests=2,
        context_file="repo_context.txt",
    )
    assert error is not None
    assert "context_file is not supported" in error["error"]
    assert workload_reference is None
    assert workload_pack is None
    assert run_plan is None
    assert support is None


@pytest.mark.asyncio
async def test_tool_get_benchmark_matrix_returns_filtered_catalog() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_get_benchmark_matrix",
        {
            "gpu_family": "blackwell",
            "model_class": "classical_moe",
            "engine": "",
        },
    )
    payload = result.structured_content

    assert payload["evidence"] == "benchmark_matrix_catalog"
    assert {descriptor["name"] for descriptor in payload["matrix"]["workloads"]} == {"kimi-k2-long-context-coding"}
    assert {descriptor["name"] for descriptor in payload["matrix"]["experiments"]} == {
        "dynamo-aggregated-lmcache-kimi-k2",
        "vllm-disagg-prefill-lmcache",
        "dynamo-disagg-lmcache-kimi-k2",
    }


@pytest.mark.asyncio
async def test_tool_list_benchmark_experiments_exposes_descriptors() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool("tool_list_benchmark_experiments")
    payload = result.structured_content
    comparison = next(
        descriptor for descriptor in payload["descriptors"] if descriptor["name"] == "vllm-disagg-prefill-lmcache"
    )

    assert payload["evidence"] == "packaged_experiment_catalog"
    assert set(payload["experiments"]) == {
        "dynamo-aggregated-lmcache-kimi-k2",
        "vllm-disagg-prefill-lmcache",
        "dynamo-disagg-lmcache-kimi-k2",
    }
    assert comparison["workload_class"] == "coding"
    assert "observability" in comparison["focus_areas"]


@pytest.mark.asyncio
async def test_tool_plan_benchmark_strategy_returns_suite() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_plan_benchmark_strategy",
        {
            "model": "Kimi-K2.5",
            "gpu": "b200",
            "workload": "coding",
            "num_gpus": 8,
            "avg_prompt_tokens": 32768,
            "has_rdma": True,
        },
    )
    payload = result.structured_content

    assert payload["evidence"] == "benchmark_strategy_planner"
    assert payload["benchmark_strategy"]["primary_workload"]["name"] == "kimi-k2-long-context-coding"
    assert [lane["experiment"] for lane in payload["benchmark_strategy"]["suite"]] == [
        "dynamo-aggregated-lmcache-kimi-k2",
        "vllm-disagg-prefill-lmcache",
        "dynamo-disagg-lmcache-kimi-k2",
    ]
    assert payload["benchmark_strategy"]["suite"][0]["support"]["gpu_isa"] == "sm_100"
    assert payload["benchmark_strategy"]["selected_engine"] == "dynamo"


@pytest.mark.asyncio
async def test_tool_resolve_benchmark_plan_rejects_unsupported_grace_lane() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_resolve_benchmark_plan",
        {
            "workload": "kimi-k2-long-context-coding",
            "endpoint": "http://localhost:8000",
            "experiment": "dynamo-disagg-lmcache-kimi-k2",
            "model": "Qwen3.5-72B",
            "gpu": "b200",
            "num_gpus": 4,
        },
    )
    payload = result.structured_content

    assert "error" in payload
    assert payload["evidence"] == "production_target_validation"
    assert "Kimi-K2.5" in payload["error"]


@pytest.mark.asyncio
async def test_tool_run_benchmark_returns_observed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(*args, **kwargs):
        del args, kwargs
        return BenchmarkArtifact(
            pack_name="kimi-k2-long-context-coding",
            workload_class="coding",
            endpoint="http://localhost:8000",
            model="Kimi-K2.5",
            concurrency=2,
            started_at="2026-03-25T00:00:00Z",
            completed_at="2026-03-25T00:00:01Z",
            run_plan={"observed_runtime": {"request_throughput_rps": 2.5}},
            results=[],
            summary=BenchmarkSummary(
                total_requests=2,
                succeeded=2,
                failed=0,
                concurrency=2,
                wall_time_ms=1000.0,
            ),
        )

    monkeypatch.setattr("inferscope.server_benchmarks.run_openai_replay", fake_run_openai_replay)
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_run_benchmark",
        {
            "workload": "kimi-k2-long-context-coding",
            "endpoint": "http://localhost:8000",
            "synthetic_requests": 2,
            "synthetic_input_tokens": 2048,
            "synthetic_output_tokens": 256,
            "gpu": "b200",
            "engine": "dynamo",
        },
    )
    payload = result.structured_content

    assert payload["observed_runtime"]["request_throughput_rps"] == 2.5
    assert payload["support"]["gpu_isa"] == "sm_100"
    assert payload["production_readiness"]["ready"] is True
