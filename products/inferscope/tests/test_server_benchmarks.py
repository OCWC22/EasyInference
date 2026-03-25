"""Tests for MCP-side benchmark resolution helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from inferscope.benchmarks import BenchmarkArtifact, BenchmarkSummary
from inferscope.server_benchmarks import _resolve_benchmark_plan, register_benchmark_tools


def test_resolve_benchmark_plan_supports_procedural_workloads() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "tool-agent",
        "http://localhost:8000",
        synthetic_requests=4,
        synthetic_input_tokens=2048,
        synthetic_output_tokens=256,
    )
    assert error is None
    assert workload_reference == "tool-agent"
    assert workload_pack is not None
    assert run_plan is not None
    assert support is not None
    assert len(workload_pack.requests) == 4
    assert run_plan.workload_ref == "tool-agent"
    assert support.status == "unknown"


def test_resolve_benchmark_plan_rejects_context_file_for_mcp() -> None:
    error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
        "coding-long-context",
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
            "gpu_family": "blackwell-grace",
            "model_class": "qwen35-hybrid",
            "focus_area": "kv_offload",
        },
    )
    payload = result.structured_content

    assert payload["evidence"] == "benchmark_matrix_catalog"
    assert {descriptor["name"] for descriptor in payload["matrix"]["workloads"]} == {"long-context-kv-offload-rag"}
    assert "vllm-disagg-prefill-lmcache-grace" in {
        descriptor["name"] for descriptor in payload["matrix"]["experiments"]
    }


@pytest.mark.asyncio
async def test_tool_list_benchmark_experiments_exposes_descriptors() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool("tool_list_benchmark_experiments")
    payload = result.structured_content
    router = next(
        descriptor for descriptor in payload["descriptors"] if descriptor["name"] == "sglang-router-prefill-decode"
    )

    assert payload["evidence"] == "packaged_experiment_catalog"
    assert router["workload_class"] == "tool_agent"
    assert "router" in router["focus_areas"]


@pytest.mark.asyncio
async def test_tool_plan_benchmark_strategy_returns_suite() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_plan_benchmark_strategy",
        {
            "model": "Qwen3.5-72B",
            "gpu": "gb200",
            "workload": "long_context_rag",
            "num_gpus": 4,
            "avg_prompt_tokens": 32768,
            "has_rdma": True,
        },
    )
    payload = result.structured_content

    assert payload["evidence"] == "benchmark_strategy_planner"
    assert payload["benchmark_strategy"]["primary_workload"]["name"] == "long-context-kv-offload-rag"
    assert payload["benchmark_strategy"]["suite"][0]["experiment"] == "vllm-single-endpoint-long-context-rag-baseline"
    assert payload["benchmark_strategy"]["suite"][0]["support"]["gpu_isa"] == "sm_100"


@pytest.mark.asyncio
async def test_tool_resolve_benchmark_plan_rejects_unsupported_grace_lane() -> None:
    mcp = FastMCP("test-benchmarks")
    register_benchmark_tools(mcp)

    result = await mcp.call_tool(
        "tool_resolve_benchmark_plan",
        {
            "workload": "long-context-kv-offload-rag",
            "endpoint": "http://localhost:8000",
            "experiment": "vllm-disagg-prefill-lmcache-grace",
            "model": "Qwen3.5-72B",
            "gpu": "h100",
            "num_gpus": 4,
        },
    )
    payload = result.structured_content

    assert "error" in payload
    assert payload["support"]["status"] == "unsupported"
    assert any(issue["code"] == "grace_tier_requires_grace" for issue in payload["support"]["issues"])


@pytest.mark.asyncio
async def test_tool_run_benchmark_returns_observed_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run_openai_replay(*args, **kwargs):
        del args, kwargs
        return BenchmarkArtifact(
            pack_name="tool-agent",
            workload_class="tool_agent",
            endpoint="http://localhost:8000",
            model="Qwen3.5-32B",
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
            "workload": "tool-agent",
            "endpoint": "http://localhost:8000",
            "synthetic_requests": 2,
            "synthetic_input_tokens": 2048,
            "synthetic_output_tokens": 256,
            "gpu": "b200",
            "engine": "sglang",
        },
    )
    payload = result.structured_content

    assert payload["observed_runtime"]["request_throughput_rps"] == 2.5
    assert payload["support"]["gpu_isa"] == "sm_100"
