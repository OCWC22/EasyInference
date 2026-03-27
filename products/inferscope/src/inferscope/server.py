"""InferScope MCP server — the main entry point for MCP-compatible agents."""

from __future__ import annotations

from fastmcp import FastMCP

from inferscope.server_benchmarks import register_benchmark_tools
from inferscope.server_profiling import register_profiling_tools
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

mcp = FastMCP(
    "inferscope",
    instructions="""InferScope is currently narrowed to one production target:
    Kimi-K2.5 on NVIDIA Dynamo for long-context coding workloads on H100, H200, B200, and B300.

    The MCP surface is intentionally biased toward reliability and observability:
    - Kimi-targeted benchmark planning for aggregated Dynamo, split vLLM, and split Dynamo LMCache lanes
    - Runtime profiling from frontend plus worker Prometheus metrics
    - Production-target contract output for MCP clients that need stable provisioning rules
    - Benchmark artifact summaries that explain what failed, what saturated, and what observability was missing

    The supported MCP contract should be treated as Kimi on Hopper/Blackwell, with Dynamo as
    the serving target and vLLM plus Dynamo as the benchmark engines, until the production lane
    expands again.
    """,
)


# === GROUP 1: HARDWARE INTELLIGENCE ===


@mcp.tool()
async def tool_get_gpu_specs(gpu: str) -> dict:
    """Return complete ISA-level specs for a GPU."""
    return get_gpu_specs(gpu)


@mcp.tool()
async def tool_compare_gpus(gpu_a: str, gpu_b: str, workload: str = "inference") -> dict:
    """Side-by-side GPU comparison with inference-relevant metrics."""
    return compare_gpus(gpu_a, gpu_b, workload)


# === GROUP 2: MODEL INTELLIGENCE ===


@mcp.tool()
async def tool_get_model_profile(model: str) -> dict:
    """Return complete serving profile for a model."""
    return get_model_profile(model)


@mcp.tool()
async def tool_validate_serving_config(
    model: str,
    gpu: str,
    tp: int = 1,
    quantization: str = "auto",
    engine: str = "dynamo",
) -> dict:
    """Pre-flight check: does this serving config work?"""
    return validate_serving_config(model, gpu, tp, quantization, engine)


@mcp.tool()
async def tool_estimate_capacity(
    model: str,
    gpu: str,
    num_gpus: int = 1,
    quantization: str = "auto",
    max_context: int = 0,
) -> dict:
    """Calculate max concurrent users, max context length, and KV cache budget."""
    return estimate_capacity(model, gpu, num_gpus, quantization, max_context)


# === GROUP 3: RECOMMENDATIONS ===


@mcp.tool()
async def tool_recommend_config(
    model: str,
    gpu: str,
    workload: str = "coding",
    num_gpus: int = 1,
    engine: str = "dynamo",
) -> dict:
    """Generate optimal serving config for a model+GPU+workload combination."""
    return recommend_config(model, gpu, workload, num_gpus, engine=engine)


@mcp.tool()
async def tool_recommend_engine(
    model: str,
    gpu: str,
    workload: str = "coding",
    num_gpus: int = 1,
    multi_node: bool = False,
) -> dict:
    """Recommend the best inference engine for this model+GPU+workload."""
    return recommend_engine(model, gpu, workload, num_gpus, multi_node)


@mcp.tool()
async def tool_suggest_parallelism(model: str, gpu: str, num_gpus: int) -> dict:
    """Recommend TP/PP/DP/EP strategy based on model architecture and hardware."""
    return suggest_parallelism(model, gpu, num_gpus)


# === GROUP 4: KV CACHE MANAGEMENT ===


@mcp.tool()
async def tool_calculate_kv_budget(
    model: str,
    context_length: int,
    batch_size: int = 1,
    kv_dtype: str = "fp8",
) -> dict:
    """Calculate exact KV cache memory requirement in bytes."""
    return calculate_kv_budget(model, context_length, batch_size, kv_dtype)


@mcp.tool()
async def tool_recommend_kv_strategy(
    model: str,
    gpu: str,
    workload: str = "coding",
    max_context: int = 32768,
    concurrent_sessions: int = 100,
) -> dict:
    """Recommend KV cache tiering strategy for this deployment."""
    return recommend_kv_strategy(model, gpu, workload, max_context, concurrent_sessions)


@mcp.tool()
async def tool_recommend_disaggregation(
    model: str,
    gpu: str,
    target_ttft_ms: float = 500.0,
    avg_prompt_tokens: int = 4096,
    request_rate_per_sec: float = 10.0,
    has_rdma: bool = False,
    num_gpus: int = 1,
) -> dict:
    """Determine if prefill/decode disaggregation would help this deployment."""
    return recommend_disaggregation(
        model,
        gpu,
        target_ttft_ms,
        avg_prompt_tokens,
        request_rate_per_sec,
        has_rdma,
        num_gpus,
    )


@mcp.tool()
async def tool_compare_quantization(model: str, gpu: str) -> dict:
    """Compare quantization options for this specific GPU and model."""
    return compare_quantization(model, gpu)


# === GROUP 5: RUNTIME PROFILING ===

register_profiling_tools(mcp)


# === GROUP 6: EVALUATION SUBSYSTEM ===

register_benchmark_tools(mcp)
