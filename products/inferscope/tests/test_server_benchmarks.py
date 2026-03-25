"""Tests for MCP-side benchmark resolution helpers."""

from inferscope.server_benchmarks import _resolve_benchmark_plan


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
