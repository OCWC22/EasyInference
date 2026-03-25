"""Regression tests for benchmark launch planning on Hopper/Blackwell."""

from __future__ import annotations

import shlex

from inferscope.benchmarks.launchers import _map_workload_mode, build_benchmark_stack_plan
from inferscope.optimization.serving_profile import WorkloadMode


def test_map_workload_mode_recognizes_rag() -> None:
    assert _map_workload_mode("rag") == WorkloadMode.LONG_CONTEXT_RAG
    assert _map_workload_mode("long_context_rag") == WorkloadMode.LONG_CONTEXT_RAG


def test_build_benchmark_stack_plan_uses_memory_valid_h200_layout_for_deepseek() -> None:
    plan = build_benchmark_stack_plan(
        "vllm-single-endpoint-baseline",
        "h200",
        8,
        model="DeepSeek-V3",
    )

    primary = next(component for component in plan.components if component.role == "primary")
    tokens = shlex.split(primary.command.replace("\\\n", " "))
    assert tokens[tokens.index("--tensor-parallel-size") + 1] == "8"


def test_build_benchmark_stack_plan_injects_offloading_connector_for_long_context_lane() -> None:
    plan = build_benchmark_stack_plan(
        "vllm-single-endpoint-offloading-connector",
        "h200",
        1,
    )

    primary = next(component for component in plan.components if component.role == "primary")
    assert "OffloadingConnector" in primary.command
    assert any("cold/idle KV offload" in note for note in primary.notes)


def test_build_benchmark_stack_plan_generates_lmcache_bundle_for_grace_lane() -> None:
    plan = build_benchmark_stack_plan(
        "vllm-disagg-prefill-lmcache-grace",
        "gb200",
        4,
    )

    generated_paths = {generated.path for generated in plan.generated_files}
    assert generated_paths == {"lmcache-prefiller-config.yaml", "lmcache-decoder-config.yaml"}
    assert not plan.warnings
    assert plan.support is not None
    assert plan.support["gpu_isa"] == "sm_100"


def test_build_benchmark_stack_plan_rejects_grace_lane_on_non_grace_gpu() -> None:
    try:
        build_benchmark_stack_plan(
            "vllm-disagg-prefill-lmcache-grace",
            "h100",
            4,
        )
    except ValueError as exc:
        assert "Grace-coherent cache tiers require" in str(exc)
    else:
        raise AssertionError("Expected Grace-only benchmark lane to be rejected on H100")
