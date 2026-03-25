"""Regression tests for Hopper/Blackwell recommendation policy."""

from __future__ import annotations

from inferscope.tools.recommend import recommend_config, recommend_engine


def test_recommend_h100_deepseek_chat_falls_back_to_memory_valid_awq_tp8() -> None:
    result = recommend_config("DeepSeek-V3", "h100", workload="chat", num_gpus=8)

    profile = result["serving_profile"]
    assert profile["engine"] == "vllm"
    assert profile["precision"]["weights"] == "awq"
    assert profile["topology"]["tp"] == 8
    assert profile["topology"]["dp"] == 1
    assert result["memory_plan"]["fits"] is True


def test_recommend_h200_deepseek_chat_uses_vllm_fp8_tp8_when_tp4_hint_does_not_fit() -> None:
    result = recommend_config("DeepSeek-V3", "h200", workload="chat", num_gpus=8)

    profile = result["serving_profile"]
    assert profile["engine"] == "vllm"
    assert profile["precision"]["weights"] == "fp8"
    assert profile["topology"]["tp"] == 8
    assert profile["topology"]["dp"] == 1
    assert profile["topology"]["ep"] == 1
    assert result["memory_plan"]["fits"] is True


def test_recommend_b200_deepseek_chat_uses_blackwell_fp4_without_grace_note() -> None:
    result = recommend_config("DeepSeek-V3", "b200", workload="chat", num_gpus=4)

    profile = result["serving_profile"]
    engine_config = result["engine_config"]
    assert profile["engine"] == "vllm"
    assert profile["precision"]["weights"] == "fp4"
    assert profile["topology"]["tp"] == 4
    assert result["memory_plan"]["fits"] is True
    assert all("Grace" not in note for note in engine_config["notes"])


def test_recommend_gb200_long_context_surfaces_grace_overflow() -> None:
    result = recommend_config("DeepSeek-V3", "gb200", workload="long_context_rag", num_gpus=4)

    engine_config = result["engine_config"]
    memory_plan = result["memory_plan"]
    assert result["serving_profile"]["engine"] == "vllm"
    assert any("Grace" in note for note in engine_config["notes"])
    assert memory_plan["platform_overflow_tier"] == "gpu_grace_coherent"
    assert memory_plan["overflow_memory_gb"] > 0
    assert memory_plan["overflow_bandwidth_gb_s"] == 546.0


def test_recommend_engine_matches_recommend_config_for_nvidia_coding_workloads() -> None:
    for gpu in ("h100", "h200", "b200", "gb200"):
        config = recommend_config("DeepSeek-V3", gpu, workload="coding", num_gpus=8 if gpu != "b200" else 4)
        engines = recommend_engine("DeepSeek-V3", gpu, workload="coding", num_gpus=8 if gpu != "b200" else 4)

        assert engines["rankings"][0]["engine"] == config["engine_config"]["engine"]
