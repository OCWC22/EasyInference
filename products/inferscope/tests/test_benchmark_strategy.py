"""Tests for benchmark strategy planning and runtime bridging."""

from __future__ import annotations

import pytest

from inferscope.benchmarks.strategy import plan_benchmark_strategy, plan_benchmark_strategy_with_runtime


def test_plan_benchmark_strategy_selects_kimi_long_context_suite() -> None:
    result = plan_benchmark_strategy(
        "Kimi-K2.5",
        "b200",
        workload="coding",
        num_gpus=8,
        avg_prompt_tokens=32768,
        concurrent_sessions=32,
        has_rdma=True,
        include_stack_plans=False,
    )

    assert result["evidence"] == "benchmark_strategy_planner"
    assert result["benchmark_strategy"]["primary_workload"]["name"] == "kimi-k2-long-context-coding"
    assert result["benchmark_strategy"]["selected_engine"] == "dynamo"
    assert [lane["experiment"] for lane in result["benchmark_strategy"]["suite"]] == [
        "dynamo-aggregated-lmcache-kimi-k2",
        "vllm-disagg-prefill-lmcache",
        "dynamo-disagg-lmcache-kimi-k2",
    ]
    assert result["benchmark_strategy"]["ready"] is True
    assert result["benchmark_strategy"]["support"]["gpu_isa"] == "sm_100"
    assert result["benchmark_strategy"]["suite"][2]["support"]["status"] == "supported"


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
        "Kimi-K2.5",
        "b200",
        workload="coding",
        num_gpus=8,
        avg_prompt_tokens=32768,
        concurrent_sessions=32,
        has_rdma=True,
        endpoint="http://localhost:8000",
        include_stack_plans=False,
    )

    assert result["evidence"] == "benchmark_strategy_runtime_bridge"
    assert result["benchmark_strategy"]["suite"][0]["experiment"] == "dynamo-disagg-lmcache-kimi-k2"
    assert captured["model_name"] == ""
    assert captured["quantization"] == ""
    assert result["benchmark_strategy"]["runtime_bridge"]["active"] is True
    assert result["benchmark_strategy"]["runtime_bridge"]["current_hints_supplied"]["model_name"] is False
    assert any("cache.offload_policy" in action for action in result["benchmark_strategy"]["next_actions"])
