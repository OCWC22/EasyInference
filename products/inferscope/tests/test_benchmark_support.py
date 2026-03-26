"""Unit tests for benchmark support assessment."""

from __future__ import annotations

from inferscope.benchmarks import assess_benchmark_support, load_experiment, load_workload


def test_assess_benchmark_support_rejects_grace_lane_on_h100() -> None:
    experiment = load_experiment("vllm-disagg-prefill-lmcache-grace")
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=4,
        engine_name="vllm",
        workload=workload,
        experiment=experiment,
        prompt_tokens=96_000,
    )

    assert support.status == "unsupported"
    assert support.gpu_isa == "sm_90a"
    assert any(issue.code == "grace_tier_requires_grace" for issue in support.issues)


def test_assess_benchmark_support_marks_nixl_transport_degraded_without_rdma() -> None:
    experiment = load_experiment("vllm-disagg-prefill-nixl")
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-32B",
        gpu_name="h100_pcie",
        num_gpus=2,
        engine_name="vllm",
        workload=workload,
        experiment=experiment,
        prompt_tokens=16_384,
        has_rdma=False,
    )

    assert support.status == "degraded"
    assert any(issue.code == "nixl_transport_degraded" for issue in support.issues)


def test_assess_benchmark_support_rejects_deprecated_simm_backend() -> None:
    """SiMM remote_backend is rejected with deprecated_remote_backend error."""
    from inferscope.benchmarks.experiments import BenchmarkExperimentSpec, BenchmarkCacheMetadata, BenchmarkTopologyMetadata

    # Construct a synthetic experiment with remote_backend="simm" to test gating
    experiment = BenchmarkExperimentSpec(
        version="1",
        name="test-simm-rejected",
        description="Synthetic SiMM experiment for gating test",
        engine="vllm",
        workload="coding-long-context",
        topology=BenchmarkTopologyMetadata(mode="prefill_decode_split"),
        cache=BenchmarkCacheMetadata(
            strategy="lmcache",
            tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
            remote_backend="simm",
        ),
    )
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=4,
        engine_name="vllm",
        workload=workload,
        experiment=experiment,
        prompt_tokens=16_384,
    )

    assert support.status == "unsupported"
    assert any(issue.code == "deprecated_remote_backend" for issue in support.issues)


def test_assess_benchmark_support_dynamo_disagg_supported() -> None:
    """Dynamo disaggregated experiment is supported on NVIDIA with multiple GPUs."""
    experiment = load_experiment("dynamo-disagg-prefill-nixl")
    workload = load_workload(experiment.workload)

    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=4,
        engine_name="dynamo",
        workload=workload,
        experiment=experiment,
        prompt_tokens=16_384,
    )

    assert support.status in {"supported", "degraded"}
    # No deprecated_remote_backend errors
    assert not any(issue.code == "deprecated_remote_backend" for issue in support.issues)


def test_assess_benchmark_support_trtllm_supported_on_nvidia() -> None:
    """TRT-LLM is now a SUPPORTED engine on NVIDIA hardware."""
    support = assess_benchmark_support(
        model_name="Qwen3.5-32B",
        gpu_name="b200",
        num_gpus=1,
        engine_name="trtllm",
        prompt_tokens=4_096,
    )

    assert support.engine_support_tier == "supported"
    # No preview_engine issue for supported engines
    assert not any(issue.code == "preview_engine" for issue in support.issues)


def test_assess_benchmark_support_dynamo_supported_single_host_multi_gpu() -> None:
    """Single-host multi-GPU should NOT promote Dynamo to RECOMMENDED."""
    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=8,
        engine_name="dynamo",
        prompt_tokens=16_384,
        multi_node=False,
    )
    # Without multi_node=True, Dynamo should be SUPPORTED, not RECOMMENDED
    assert support.engine_support_tier == "supported"


def test_assess_benchmark_support_dynamo_recommended_multi_node() -> None:
    """Explicit multi_node=True should promote Dynamo to RECOMMENDED."""
    support = assess_benchmark_support(
        model_name="Qwen3.5-72B",
        gpu_name="h100",
        num_gpus=8,
        engine_name="dynamo",
        prompt_tokens=16_384,
        multi_node=True,
    )
    assert support.engine_support_tier == "recommended"
