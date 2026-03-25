"""CLI tests for benchmark catalog and matrix commands."""

from __future__ import annotations

from typer.testing import CliRunner

from inferscope.cli import app

runner = CliRunner()


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
