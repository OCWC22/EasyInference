"""InferScope CLI — standalone command-line interface.

Usage:
    inferscope profile DeepSeek-R1
    inferscope profile-runtime http://localhost:8000
    inferscope validate DeepSeek-R1 h100 --tp 8
    inferscope recommend DeepSeek-R1 mi355x --num-gpus 8 --workload coding
    inferscope gpu h100
    inferscope compare h100 mi355x
    inferscope capacity Llama-3-70B h200 --num-gpus 1
    inferscope serve  # Start MCP server (stdio)
"""

from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

from inferscope.cli_benchmarks import register_benchmark_commands
from inferscope.cli_profiling import register_profiling_commands
from inferscope.endpoint_auth import parse_header_values, resolve_auth_config
from inferscope.tools.hardware_intel import compare_gpus, get_gpu_specs
from inferscope.tools.kv_cache import (
    calculate_kv_budget,
    compare_quantization,
    recommend_disaggregation,
    recommend_kv_strategy,
)
from inferscope.tools.model_intel import (
    estimate_capacity,
    get_model_profile,
    validate_serving_config,
)
from inferscope.tools.recommend import (
    recommend_config,
    recommend_engine,
    suggest_parallelism,
)

app = typer.Typer(
    name="inferscope",
    help="Hardware-aware inference optimization for LLM serving engines.",
    no_args_is_help=True,
)
console = Console()


def _print_result(result: dict) -> None:
    """Pretty-print a tool result."""
    summary = result.pop("summary", "")
    confidence = result.pop("confidence", None)

    if summary:
        console.print(f"\n[bold]{summary}[/bold]")

    if confidence is not None:
        level = "green" if confidence >= 0.8 else "yellow" if confidence >= 0.6 else "red"
        console.print(f"Confidence: [{level}]{confidence:.0%}[/{level}]")

    launch_cmd = result.pop("launch_command", None)
    if launch_cmd:
        console.print("\n[bold cyan]Launch command:[/bold cyan]")
        console.print(Panel(Syntax(launch_cmd, "bash", theme="monokai"), border_style="cyan"))

    output = json.dumps(result, indent=2, default=str)
    console.print(Syntax(output, "json", theme="monokai"))


def _resolve_metrics_auth(
    *,
    provider: str = "",
    metrics_api_key: str = "",
    metrics_auth_scheme: str = "",
    metrics_auth_header_name: str = "",
    metrics_header: list[str] | None = None,
):
    try:
        return resolve_auth_config(
            (metrics_api_key or None),
            provider=provider,
            auth_scheme=metrics_auth_scheme,
            auth_header_name=metrics_auth_header_name,
            headers=parse_header_values(metrics_header, option_name="metrics header"),
            default_scheme="bearer",
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def profile(model: str = typer.Argument(help="Model name (e.g., DeepSeek-R1, Qwen3.5-72B)")):
    """Show serving profile for a model across all supported engines and GPUs."""
    _print_result(get_model_profile(model))


@app.command()
def validate(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type (e.g., h100, mi355x)"),
    tp: int = typer.Option(1, help="Tensor parallelism degree"),
    quantization: str = typer.Option("auto", help="Quantization (fp8, bf16, awq, etc.)"),
    engine: str = typer.Option("vllm", help="Engine (vllm, sglang, atom)"),
):
    """Validate a serving config before deployment."""
    _print_result(validate_serving_config(model, gpu, tp, quantization, engine))


@app.command(name="recommend")
def recommend_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload: coding, chat, agent, long_context_rag"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    engine: str = typer.Option("auto", help="Engine (auto, vllm, sglang, atom, trtllm, dynamo)"),
):
    """Generate optimal ServingProfile for this deployment."""
    _print_result(recommend_config(model, gpu, workload, num_gpus, engine=engine))


@app.command()
def gpu(name: str = typer.Argument(help="GPU name (e.g., h100, a100, mi355x)")):
    """Show complete ISA-level specs for a GPU."""
    _print_result(get_gpu_specs(name))


@app.command()
def compare(
    gpu_a: str = typer.Argument(help="First GPU"),
    gpu_b: str = typer.Argument(help="Second GPU"),
):
    """Side-by-side GPU comparison with inference-relevant metrics."""
    _print_result(compare_gpus(gpu_a, gpu_b))


@app.command()
def capacity(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    quantization: str = typer.Option("auto", help="Quantization method"),
):
    """Calculate max concurrent users and KV cache budget."""
    _print_result(estimate_capacity(model, gpu, num_gpus, quantization))


@app.command(name="engine")
def engine_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload type (coding, chat, agent, long_context_rag)"),
    num_gpus: int = typer.Option(1, help="Number of GPUs"),
    multi_node: bool = typer.Option(False, help="Multi-node deployment"),
):
    """Recommend the best inference engine for this deployment."""
    _print_result(recommend_engine(model, gpu, workload, num_gpus, multi_node))


@app.command()
def parallelism(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    num_gpus: int = typer.Argument(help="Number of GPUs available"),
):
    """Recommend TP/PP/DP/EP parallelism strategy."""
    _print_result(suggest_parallelism(model, gpu, num_gpus))


@app.command(name="kv-budget")
def kv_budget(
    model: str = typer.Argument(help="Model name"),
    context_length: int = typer.Argument(help="Context length in tokens"),
    batch_size: int = typer.Option(1, help="Batch size"),
    kv_dtype: str = typer.Option("fp8", help="KV cache dtype (fp8, fp16)"),
):
    """Calculate exact KV cache memory requirement."""
    _print_result(calculate_kv_budget(model, context_length, batch_size, kv_dtype))


@app.command(name="kv-strategy")
def kv_strategy(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    workload: str = typer.Option("chat", help="Workload type (coding, chat, agent, long_context_rag)"),
    max_context: int = typer.Option(32768, help="Max context length"),
    concurrent_sessions: int = typer.Option(100, help="Concurrent sessions"),
):
    """Recommend KV cache tiering strategy."""
    _print_result(recommend_kv_strategy(model, gpu, workload, max_context, concurrent_sessions))


@app.command(name="disagg")
def disagg(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
    avg_prompt: int = typer.Option(4096, help="Average prompt tokens"),
    rate: float = typer.Option(10.0, help="Requests per second"),
    rdma: bool = typer.Option(False, help="RDMA available"),
    num_gpus: int = typer.Option(2, help="Number of GPUs"),
):
    """Determine if prefill/decode disaggregation would help."""
    _print_result(recommend_disaggregation(model, gpu, 500.0, avg_prompt, rate, rdma, num_gpus))


@app.command(name="quantization")
def quantization_cmd(
    model: str = typer.Argument(help="Model name"),
    gpu: str = typer.Argument(help="GPU type"),
):
    """Compare quantization options for this model + GPU."""
    _print_result(compare_quantization(model, gpu))


register_profiling_commands(app, print_result=_print_result, resolve_metrics_auth=_resolve_metrics_auth)
register_benchmark_commands(app, print_result=_print_result)


@app.command()
def serve(
    transport: str = typer.Option("stdio", help="Transport: stdio or streamable-http"),
    port: int = typer.Option(8765, help="Port for HTTP transport"),
):
    """Start the InferScope MCP server."""
    from inferscope.server import mcp

    console.print(f"[bold green]Starting InferScope MCP server ({transport})...[/bold green]")
    if transport == "stdio":
        mcp.run(transport="stdio")
    else:
        console.print(
            "[yellow]⚠ HTTP transport binds to 127.0.0.1 (localhost only). "
            "Use a reverse proxy with authentication for production.[/yellow]"
        )
        mcp.run(transport="streamable-http", host="127.0.0.1", port=port)


if __name__ == "__main__":
    app()
