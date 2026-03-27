"""CLI tests for benchmark catalog and matrix commands."""

from __future__ import annotations

from typer.testing import CliRunner

from inferscope.cli import app

runner = CliRunner()


def test_cli_benchmark_strategy_command_exists_and_runs(monkeypatch) -> None:
    async def fake_plan_strategy(*args, **kwargs):
        del args, kwargs
        return {
            "summary": "Benchmark strategy ready",
            "confidence": 0.9,
            "benchmark_strategy": {
                "suite": [
                    {"experiment": "vllm-single-endpoint-long-context-rag-baseline"},
                    {"experiment": "vllm-single-endpoint-offloading-connector"},
                ]
            },
        }

    monkeypatch.setattr("inferscope.cli_benchmarks.plan_benchmark_strategy_with_runtime", fake_plan_strategy)

    result = runner.invoke(
        app,
        [
            "benchmark-strategy",
            "Qwen3.5-72B",
            "gb200",
            "--workload",
            "long_context_rag",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "Benchmark strategy ready" in result.stdout
    assert "vllm-single-endpoint-offloading-connector" in result.stdout


def test_cli_benchmark_matrix_command_filters_long_context_offload_lane() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-matrix",
            "--gpu-family",
            "blackwell-grace",
            "--model-class",
            "qwen35-hybrid",
            "--focus-area",
            "kv_offload",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "long-context-kv-offload-rag" in result.stdout
    assert "vllm-disagg-prefill-lmcache-grace" in result.stdout


def test_cli_benchmark_workloads_lists_descriptor_metadata() -> None:
    result = runner.invoke(app, ["benchmark-workloads"])

    assert result.exit_code == 0, result.stdout
    assert "benchmark_role" in result.stdout
    assert "tool-agent" in result.stdout


def test_cli_benchmark_plan_includes_support_and_gpu_isa() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-plan",
            "kimi-k2-long-context-coding",
            "http://localhost:8000",
            "--experiment",
            "dynamo-aggregated-lmcache-kimi-k2",
            "--model",
            "Kimi-K2.5",
            "--gpu",
            "b200",
            "--num-gpus",
            "4",
            "--engine",
            "dynamo",
            "--metrics-target",
            "frontend=http://localhost:9100",
            "--metrics-target",
            "worker=http://localhost:9200",
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert '"support"' in result.stdout
    assert '"gpu_isa": "sm_100"' in result.stdout


def test_cli_benchmark_stack_plan_rejects_unsupported_grace_lane_on_h100() -> None:
    result = runner.invoke(
        app,
        [
            "benchmark-stack-plan",
            "vllm-disagg-prefill-lmcache-grace",
            "h100",
            "--num-gpus",
            "4",
        ],
    )

    assert result.exit_code != 0
    assert "Grace-coherent cache tiers require" in (result.stdout + result.stderr)
