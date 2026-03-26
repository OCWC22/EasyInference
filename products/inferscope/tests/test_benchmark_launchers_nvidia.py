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


def test_build_benchmark_stack_plan_dynamo_disagg_resolves_all_metrics_targets() -> None:
    """Dynamo disaggregated stack plan must resolve primary + router + prefill + decode targets."""
    plan = build_benchmark_stack_plan(
        "dynamo-disagg-prefill-nixl",
        "h100",
        4,
    )

    # Components: dynamo-router, vllm-prefill, vllm-decode
    component_names = {c.name for c in plan.components}
    assert "dynamo-router" in component_names
    assert "vllm-prefill" in component_names
    assert "vllm-decode" in component_names

    # Run plan must have resolved metrics targets
    run_plan = plan.run_plan
    assert run_plan is not None
    metrics_targets = run_plan.get("metrics_targets", [])
    target_names = {t["name"] for t in metrics_targets}
    assert "primary" in target_names
    assert "router" in target_names
    assert "prefill" in target_names
    assert "decode" in target_names

    # Primary target should be the decode worker (vLLM), not the router
    primary_target = next(t for t in metrics_targets if t["name"] == "primary")
    assert primary_target["expected_engine"] == "vllm"

    # Router target is optional (required=false) and expects dynamo
    router_target = next(t for t in metrics_targets if t["name"] == "router")
    assert router_target["expected_engine"] == "dynamo"

    # Worker targets expect vllm
    prefill_target = next(t for t in metrics_targets if t["name"] == "prefill")
    decode_target = next(t for t in metrics_targets if t["name"] == "decode")
    assert prefill_target["expected_engine"] == "vllm"
    assert decode_target["expected_engine"] == "vllm"

    # Generated files should include Dynamo config
    generated_paths = {g.path for g in plan.generated_files}
    assert "configs/dynamo-config.yaml" in generated_paths

    # Benchmark command must include metrics-target flags
    assert "--metrics-target" in plan.benchmark_command

    # Support should resolve ISA
    assert plan.support is not None
    assert plan.support["gpu_isa"] == "sm_90a"


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
