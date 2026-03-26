"""Tests for benchmark strategy planning and runtime bridging."""

from __future__ import annotations

import pytest

from inferscope.benchmarks.strategy import plan_benchmark_strategy, plan_benchmark_strategy_with_runtime


def test_plan_benchmark_strategy_selects_long_context_rag_suite_on_grace() -> None:
    result = plan_benchmark_strategy(
        "Qwen3.5-72B",
        "gb200",
        workload="long_context_rag",
        num_gpus=4,
        avg_prompt_tokens=32768,
        concurrent_sessions=32,
        has_rdma=True,
        include_stack_plans=False,
    )

    assert result["evidence"] == "benchmark_strategy_planner"
    assert result["benchmark_strategy"]["primary_workload"]["name"] == "long-context-kv-offload-rag"
    assert result["benchmark_strategy"]["selected_engine"] == "vllm"
    suite_experiments = [lane["experiment"] for lane in result["benchmark_strategy"]["suite"]]
    # Core lanes must be present in order
    assert suite_experiments[:3] == [
        "vllm-single-endpoint-long-context-rag-baseline",
        "vllm-single-endpoint-offloading-connector",
        "dynamo-disagg-prefill-nixl-grace",
    ]
    # LMCache comparison lane follows the Dynamo primary lane
    assert "vllm-disagg-prefill-lmcache-grace" in suite_experiments
    assert result["benchmark_strategy"]["suite"][2]["required"] is True
    assert result["benchmark_strategy"]["ready"] is True
    assert result["benchmark_strategy"]["support"]["gpu_isa"] == "sm_100"
    assert result["benchmark_strategy"]["suite"][2]["support"]["status"] == "supported"


def test_plan_benchmark_strategy_coding_selects_dynamo_on_nvidia_multi_gpu() -> None:
    result = plan_benchmark_strategy(
        "Qwen3.5-72B",
        "h100",
        workload="coding",
        num_gpus=4,
        has_rdma=True,
        include_stack_plans=False,
    )

    suite_experiments = [lane["experiment"] for lane in result["benchmark_strategy"]["suite"]]
    # Dynamo should appear as a disaggregated lane for NVIDIA coding
    assert "dynamo-disagg-prefill-nixl" in suite_experiments
    # Reference baseline must be first
    assert suite_experiments[0] == "vllm-single-endpoint-baseline"


@pytest.mark.asyncio
async def test_plan_benchmark_strategy_with_runtime_prioritizes_cache_pressure(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    async def fake_profile_runtime(*args, **kwargs):
        del args
        captured.update(kwargs)
        return {
            "memory_pressure": {"level": "critical"},
            "cache_effectiveness": {"effectiveness": "poor"},
            "bottlenecks": [{"kind": "cache_bound"}],
            "tuning_preview": {
                "adjustments": [
                    {
                        "parameter": "cache.offload_policy",
                        "recommended_value": "disabled",
                    }
                ]
            },
            "summary": "runtime ok",
            "confidence": 0.9,
        }

    monkeypatch.setattr("inferscope.benchmarks.strategy.profile_runtime", fake_profile_runtime)

    result = await plan_benchmark_strategy_with_runtime(
        "Qwen3.5-72B",
        "gb200",
        workload="long_context_rag",
        num_gpus=4,
        avg_prompt_tokens=32768,
        concurrent_sessions=32,
        has_rdma=True,
        endpoint="http://localhost:8000",
        include_stack_plans=False,
    )

    assert result["evidence"] == "benchmark_strategy_runtime_bridge"
    assert result["benchmark_strategy"]["suite"][0]["phase"] == "offload"
    assert captured["model_name"] == ""
    assert captured["quantization"] == ""
    assert result["benchmark_strategy"]["runtime_bridge"]["active"] is True
    assert result["benchmark_strategy"]["runtime_bridge"]["current_hints_supplied"]["model_name"] is False
    assert any("cache.offload_policy" in action for action in result["benchmark_strategy"]["next_actions"])
