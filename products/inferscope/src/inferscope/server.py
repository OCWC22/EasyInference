"""InferScope MCP server — the main entry point for MCP-compatible agents.

Supports stdio transport (Claude Code, Codex, Cursor) and
Streamable HTTP (production / remote agents).
"""

from __future__ import annotations

from fastmcp import FastMCP

from inferscope.endpoint_auth import resolve_auth_payload
from inferscope.server_benchmarks import register_benchmark_tools
from inferscope.tools.audit import audit_deployment
from inferscope.tools.diagnose import (
    check_deployment,
    check_memory_pressure,
    get_cache_effectiveness,
)
from inferscope.tools.hardware_intel import compare_gpus, get_gpu_specs
from inferscope.tools.kv_cache import (
    calculate_kv_budget,
    compare_quantization,
    recommend_disaggregation,
    recommend_kv_strategy,
)
from inferscope.tools.live_tuner import auto_tune_deployment
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
    instructions="""InferScope is a hardware-aware inference optimization toolkit for
    LLM serving on NVIDIA (Ampere/Hopper/Blackwell) and AMD (CDNA3/CDNA4) GPUs.

    All recommendations are grounded in ISA-level hardware knowledge — not generic advice.
    Recommendations produce exact engine-specific launch commands.

    Core product capabilities:
    - recommend_config: Optimal serving config per model + GPU + workload (with launch command)
    - validate_serving_config: Pre-flight checks (TP divisibility, memory fit, format compat)
    - recommend_engine: Best engine per deployment scenario
    - suggest_parallelism: TP/PP/DP/EP strategy per model + hardware
    - estimate_capacity: Max concurrent users, context length, KV cache budget
    - recommend_kv_strategy: KV cache tiering and connector selection
    - recommend_disaggregation: Prefill/decode split decision with connector recommendation
    - compare_quantization: FP16/FP8/FP4/INT8/AWQ options ranked per GPU
    - get_gpu_specs / compare_gpus: ISA-level GPU hardware reference
    - get_model_profile: Model architecture, memory requirements, serving commands

    Evaluation subsystem capabilities:
    - packaged workload and experiment catalogs
    - benchmark plan resolution and OpenAI-compatible replay
    - artifact comparison and benchmark stack planning/materialization

    Profiling is currently exposed as an advisory seam (nsys / rocprofv3 intent).
    Direct kernel execution remains future work.
    """,
)


# === GROUP 1: HARDWARE INTELLIGENCE ===


@mcp.tool()
async def tool_get_gpu_specs(gpu: str) -> dict:
    """Return complete ISA-level specs for a GPU.

    Includes tensor core instructions, memory hierarchy, cache sizes,
    FP8/FP4 support, interconnect bandwidth, and inference-specific notes.

    Examples: h100, a100, b200, mi300x, mi355x, a10g
    """
    return get_gpu_specs(gpu)


@mcp.tool()
async def tool_compare_gpus(gpu_a: str, gpu_b: str, workload: str = "inference") -> dict:
    """Side-by-side GPU comparison with inference-relevant metrics.

    Includes roofline analysis and cost-performance ratio.
    Examples: compare_gpus("h100", "mi355x") or compare_gpus("a100", "h200")
    """
    return compare_gpus(gpu_a, gpu_b, workload)


# === GROUP 2: MODEL INTELLIGENCE ===


@mcp.tool()
async def tool_get_model_profile(model: str) -> dict:
    """Return complete serving profile for a model.

    Architecture (dense/MoE/MLA), memory requirements, recommended GPUs,
    exact vLLM/SGLang/ATOM commands, known issues, and optimization tips.

    Examples: DeepSeek-R1, Qwen3.5-72B, Kimi-K2.5, Mixtral-8x7B, Llama-3-70B
    """
    return get_model_profile(model)


@mcp.tool()
async def tool_validate_serving_config(
    model: str,
    gpu: str,
    tp: int = 1,
    quantization: str = "auto",
    engine: str = "vllm",
) -> dict:
    """Pre-flight check: does this serving config work?

    Checks TP divisibility, memory fit, format compatibility, known bugs.
    Run this BEFORE deploying to catch misconfigurations early.
    """
    return validate_serving_config(model, gpu, tp, quantization, engine)


@mcp.tool()
async def tool_estimate_capacity(
    model: str,
    gpu: str,
    num_gpus: int = 1,
    quantization: str = "auto",
    max_context: int = 0,
) -> dict:
    """Calculate max concurrent users, max context length, and KV cache budget.

    Uses exact per-layer KV cache formulas from model profile.
    """
    return estimate_capacity(model, gpu, num_gpus, quantization, max_context)


# === GROUP 3: RECOMMENDATIONS ===


@mcp.tool()
async def tool_recommend_config(
    model: str,
    gpu: str,
    workload: str = "chat",
    num_gpus: int = 1,
    engine: str = "auto",
) -> dict:
    """Generate optimal serving config for a model+GPU+workload combination."""
    return recommend_config(model, gpu, workload, num_gpus, engine=engine)


@mcp.tool()
async def tool_recommend_engine(
    model: str,
    gpu: str,
    workload: str = "chat",
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
    workload: str = "chat",
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


# === GROUP 5: LIVE DIAGNOSTICS ===


@mcp.tool()
async def tool_audit_deployment(
    endpoint: str,
    gpu_arch: str = "",
    model_name: str = "",
    model_type: str = "",
    attention_type: str = "",
    experts_total: int = 0,
    tp: int = 1,
    quantization: str = "",
    kv_cache_dtype: str = "",
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict:
    """Run all audit checks against a live vLLM/SGLang/ATOM endpoint."""
    return await audit_deployment(
        endpoint,
        gpu_arch=gpu_arch,
        model_name=model_name,
        model_type=model_type,
        attention_type=attention_type,
        experts_total=experts_total,
        tp=tp,
        quantization=quantization,
        kv_cache_dtype=kv_cache_dtype,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
    )


@mcp.tool()
async def tool_check_deployment(
    endpoint: str,
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict:
    """Scrape a live endpoint and return a health snapshot."""
    return await check_deployment(
        endpoint,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
    )


@mcp.tool()
async def tool_check_memory_pressure(
    endpoint: str,
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict:
    """Analyze KV cache utilization and preemption rates from live metrics."""
    return await check_memory_pressure(
        endpoint,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
    )


@mcp.tool()
async def tool_get_cache_effectiveness(
    endpoint: str,
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict:
    """Measure prefix cache hit rate and cache-aware routing effectiveness."""
    return await get_cache_effectiveness(
        endpoint,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
    )


@mcp.tool()
async def tool_auto_tune_deployment(
    endpoint: str,
    current_engine: str = "",
    current_workload: str = "",
    current_scheduler: dict | None = None,
    current_cache: dict | None = None,
    provider: str = "",
    metrics_auth: dict | None = None,
) -> dict:
    """Analyze a live endpoint and recommend config adjustments."""
    return await auto_tune_deployment(
        endpoint,
        current_engine=current_engine,
        current_workload=current_workload,
        current_scheduler=current_scheduler,
        current_cache=current_cache,
        allow_private=False,
        metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
    )


# === GROUP 6: EVALUATION SUBSYSTEM ===

register_benchmark_tools(mcp)
