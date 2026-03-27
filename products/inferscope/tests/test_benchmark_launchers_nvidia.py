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
        "dynamo-aggregated-lmcache-kimi-k2",
        "h200",
        8,
        model="Kimi-K2.5",
    )

    worker = next(component for component in plan.components if component.name == "dynamo-worker")
    tokens = shlex.split(worker.command)
    assert "--model" in tokens
    assert worker.env_vars["DYN_SYSTEM_PORT"] == "8081"


def test_build_benchmark_stack_plan_injects_offloading_connector_for_long_context_lane() -> None:
    plan = build_benchmark_stack_plan(
        "dynamo-aggregated-lmcache-kimi-k2",
        "h100",
        8,
    )

    frontend = next(component for component in plan.components if component.name == "dynamo-frontend")
    assert "--router-mode kv" in frontend.command
    assert frontend.metrics_endpoint == "http://127.0.0.1:8000"


def test_build_benchmark_stack_plan_generates_dynamo_disagg_bundle_for_kimi() -> None:
    plan = build_benchmark_stack_plan(
        "dynamo-disagg-lmcache-kimi-k2",
        "b200",
        8,
    )

    component_names = {component.name for component in plan.components}
    prefill = next(component for component in plan.components if component.name == "dynamo-prefill")
    decode = next(component for component in plan.components if component.name == "dynamo-decode")
    assert component_names == {"dynamo-frontend", "dynamo-decode", "dynamo-prefill"}
    assert len(prefill.gpu_ids) >= 2
    assert len(decode.gpu_ids) >= 2
    assert len(prefill.gpu_ids) + len(decode.gpu_ids) == 8
    assert not plan.warnings
    assert plan.support is not None
    assert plan.support["gpu_isa"] == "sm_100"


def test_build_benchmark_stack_plan_generates_vllm_disagg_bundle_for_kimi() -> None:
    plan = build_benchmark_stack_plan(
        "vllm-disagg-prefill-lmcache",
        "b200",
        8,
        model="Kimi-K2.5",
    )

    component_names = {component.name for component in plan.components}
    assert component_names == {"vllm-prefill", "vllm-decode", "vllm-disagg-proxy"}
    assert len(plan.generated_files) == 2
    assert plan.support is not None
    assert plan.support["gpu_isa"] == "sm_100"


def test_build_benchmark_stack_plan_rejects_non_target_blackwell_grace_gpu() -> None:
    try:
        build_benchmark_stack_plan(
            "dynamo-disagg-lmcache-kimi-k2",
            "gb200",
            8,
        )
    except ValueError as exc:
        assert "H100, H200, B200, B300" in str(exc)
    else:
        raise AssertionError("Expected unsupported Grace GPU family to be rejected")
