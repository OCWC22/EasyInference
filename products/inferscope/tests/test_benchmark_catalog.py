"""Tests for benchmark catalog descriptors and matrix filtering."""

from __future__ import annotations

from inferscope.benchmarks.catalog import (
    build_benchmark_matrix,
    describe_builtin_experiments,
    describe_builtin_workloads,
)


def test_describe_builtin_workloads_includes_matrix_metadata() -> None:
    descriptors = describe_builtin_workloads()

    coding = next(descriptor for descriptor in descriptors if descriptor["name"] == "coding-long-context")

    assert coding["benchmark_role"] == "reference"
    assert "blackwell_grace" in coding["target_gpu_families"]
    assert "qwen35_hybrid" in coding["target_model_classes"]
    assert "prefix_cache" in coding["focus_areas"]
    assert coding["procedural"] is True


def test_describe_builtin_experiments_includes_workload_class_and_metadata() -> None:
    descriptors = describe_builtin_experiments()

    grace = next(descriptor for descriptor in descriptors if descriptor["name"] == "vllm-disagg-prefill-lmcache-grace")

    assert grace["workload"] == "long-context-kv-offload-rag"
    assert grace["workload_class"] == "long_context_rag"
    assert grace["benchmark_role"] == "topology_probe"
    assert "blackwell_grace" in grace["target_gpu_families"]
    assert "kv_offload" in grace["focus_areas"]
    assert grace["cache_strategy"] == "lmcache"


def test_build_benchmark_matrix_filters_kv_offload_grace_lane() -> None:
    matrix = build_benchmark_matrix(
        gpu_family="blackwell-grace",
        model_class="qwen35-hybrid",
        focus_area="kv_offload",
    )

    workload_names = {descriptor["name"] for descriptor in matrix["workloads"]}
    experiment_names = {descriptor["name"] for descriptor in matrix["experiments"]}
    suggested_pairs = {(pair["experiment"], pair["workload"]) for pair in matrix["suggested_pairs"]}

    assert workload_names == {"long-context-kv-offload-rag"}
    assert "vllm-single-endpoint-offloading-connector" in experiment_names
    assert "vllm-disagg-prefill-lmcache-grace" in experiment_names
    assert ("vllm-disagg-prefill-lmcache-grace", "long-context-kv-offload-rag") in suggested_pairs


def test_build_benchmark_matrix_filters_tool_agent_sglang_lane() -> None:
    matrix = build_benchmark_matrix(
        workload_class="tool-agent",
        focus_area="tool_calling",
        engine="sglang",
    )

    workload_names = {descriptor["name"] for descriptor in matrix["workloads"]}
    experiment_names = {descriptor["name"] for descriptor in matrix["experiments"]}

    assert workload_names == {"tool-agent"}
    assert experiment_names == {"sglang-router-prefill-decode", "sglang-single-endpoint-hicache"}
