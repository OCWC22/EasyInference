"""Tests for MCP-side benchmark resolution helpers."""

from __future__ import annotations

import pytest
from fastmcp import FastMCP

from inferscope.server_benchmarks import _resolve_benchmark_plan, register_benchmark_tools


def test_resolve_benchmark_plan_supports_procedural_workloads() -> None:
    error, workload_reference, workload_pack, run_plan = _resolve_benchmark_plan(
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
    assert len(workload_pack.requests) == 4
    assert run_plan.workload_ref == "tool-agent"


def test_resolve_benchmark_plan_rejects_context_file_for_mcp() -> None:
    error, workload_reference, workload_pack, run_plan = _resolve_benchmark_plan(
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
