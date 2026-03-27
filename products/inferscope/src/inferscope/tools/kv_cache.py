"""KV cache management tools for the Dynamo + LMCache production lane."""

from __future__ import annotations

from typing import Any

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.memory_planner import plan_memory
from inferscope.optimization.platform_policy import resolve_platform_traits
from inferscope.optimization.target_profile import (
    is_target_gpu,
    is_target_model,
    normalize_target_workload_class,
    supported_gpu_aliases,
    supported_model_names,
    target_profile_summary,
)


def _resolve_supported_model(model: str) -> tuple[Any, dict[str, Any] | None]:
    variant = get_model_variant(model)
    if variant is None or not is_target_model(variant):
        return None, {
            "error": f"Unsupported model: '{model}'",
            "available_models": supported_model_names(),
            "summary": target_profile_summary(),
            "confidence": 0.0,
        }
    return variant, None


def _resolve_supported_gpu(gpu: str) -> tuple[Any, dict[str, Any] | None]:
    profile = get_gpu_profile(gpu)
    if profile is None or not is_target_gpu(profile):
        return None, {
            "error": f"Unsupported GPU: '{gpu}'",
            "available_gpus": supported_gpu_aliases(),
            "summary": target_profile_summary(),
            "confidence": 0.0,
        }
    return profile, None


def calculate_kv_budget(
    model: str,
    context_length: int,
    batch_size: int = 1,
    kv_dtype: str = "fp8",
) -> dict:
    """Calculate exact KV cache memory requirement in bytes."""
    variant, error = _resolve_supported_model(model)
    if error is not None:
        return error

    kv_per_token_per_layer = variant.kv_cache_bytes_per_token(kv_dtype)
    kv_per_token_all_layers = kv_per_token_per_layer * variant.layers
    kv_per_sequence = kv_per_token_all_layers * context_length
    kv_total = kv_per_sequence * batch_size
    kv_total_gb = kv_total / (1024**3)

    return {
        "kv_budget": {
            "kv_per_token_bytes": round(kv_per_token_all_layers, 1),
            "kv_per_sequence_bytes": round(kv_per_sequence, 0),
            "kv_per_sequence_mb": round(kv_per_sequence / (1024**2), 2),
            "kv_total_bytes": round(kv_total, 0),
            "kv_total_gb": round(kv_total_gb, 3),
            "context_length": context_length,
            "batch_size": batch_size,
            "kv_dtype": kv_dtype,
        },
        "reliability_notes": [
            "Treat KV budget as the active long-context working set, not total LMCache capacity.",
            "For MCP reliability, keep enough HBM headroom for prompt bursts and cache rehydration.",
        ],
        "model": variant.name,
        "target_profile": "dynamo_long_context_coding",
        "summary": (
            f"{variant.name} @ {context_length // 1024}K × {batch_size} ({kv_dtype}) "
            f"requires {kv_total_gb:.2f} GB of active KV."
        ),
        "confidence": 0.95,
        "evidence": "architecture_based_calculation",
    }


def recommend_kv_strategy(
    model: str,
    gpu: str,
    workload: str = "coding",
    max_context: int = 32768,
    concurrent_sessions: int = 100,
) -> dict:
    """Recommend the Dynamo + LMCache KV strategy for the supported deployment."""
    variant, error = _resolve_supported_model(model)
    if error is not None:
        return error
    gpu_profile, error = _resolve_supported_gpu(gpu)
    if error is not None:
        return error

    normalized_workload = normalize_target_workload_class(workload)
    if normalized_workload != "coding":
        return {
            "error": "Supported workload is coding.",
            "summary": target_profile_summary(),
            "confidence": 0.0,
        }

    traits = resolve_platform_traits(gpu_profile)
    kv_dtype = "fp8" if gpu_profile.fp8_support else "fp16"
    kv_per_token = variant.kv_cache_bytes_per_token(kv_dtype) * variant.layers
    kv_per_session = kv_per_token * max_context
    total_kv_gb = (kv_per_session * concurrent_sessions * 1.20) / (1024**3)

    mem = plan_memory(
        model=variant,
        gpu=gpu_profile,
        num_gpus=1,
        tp=1,
        precision="fp4" if variant.name == "Kimi-K2.5" and gpu_profile.fp4_support else "fp8",
        kv_precision="fp8_e4m3",
    )
    gpu_kv_budget_gb = mem.kv_cache_budget_gb if mem.fits else 0.0
    disagg_recommended = concurrent_sessions >= 64 or total_kv_gb > max(gpu_kv_budget_gb, 1.0)
    topology = "prefill_decode_split" if disagg_recommended else "single_endpoint"
    lmcache_mode = "shared" if disagg_recommended else "local"

    strategy: dict[str, Any] = {
        "topology": topology,
        "cache_backend": "lmcache",
        "lmcache_mode": lmcache_mode,
        "tiers": ["gpu_hbm"] + (["cpu_dram"] if disagg_recommended else []),
        "connector": "LMCache shared namespace" if disagg_recommended else "LMCache local namespace",
        "session_affinity": True,
        "observability_targets": ["primary", "cache"] + (["prefill", "decode"] if disagg_recommended else []),
        "notes": [
            "Sticky session routing is mandatory for coding-session prefix reuse.",
            (
                "Cache metrics must be captured alongside request latency so MCP clients "
                "can explain reliability regressions."
            ),
        ],
    }

    if disagg_recommended:
        strategy["notes"].append(
            "Use shared LMCache for aggregate/disaggregate Dynamo serving so prefill reuse "
            "and decode reliability can be observed independently."
        )
        if not (traits.has_high_speed_interconnect):
            strategy["notes"].append(
                "Transport is the likely bottleneck: validate RDMA or tune concurrency "
                "carefully before production disaggregation."
            )
    else:
        strategy["notes"].append(
            "Single-endpoint local LMCache is the default production lane when HBM "
            "headroom and session counts stay bounded."
        )

    return {
        "strategy": strategy,
        "kv_budget": {
            "per_session_mb": round(kv_per_session / (1024**2), 2),
            "total_gb": round(total_kv_gb, 2),
            "gpu_kv_budget_gb": round(gpu_kv_budget_gb, 2),
            "fits_in_gpu": total_kv_gb <= gpu_kv_budget_gb,
            "platform_overflow_tier": mem.platform_overflow_tier,
        },
        "model": variant.name,
        "gpu": gpu_profile.name,
        "workload": "coding",
        "target_profile": "dynamo_long_context_coding",
        "summary": (
            f"{variant.name} on {gpu_profile.name}: use Dynamo with {lmcache_mode} LMCache "
            f"and {topology} topology for {concurrent_sessions} coding sessions @ "
            f"{max_context // 1024}K context."
        ),
        "confidence": 0.9,
        "evidence": "lmcache_strategy_policy",
    }


def recommend_disaggregation(
    model: str,
    gpu: str,
    target_ttft_ms: float = 500.0,
    avg_prompt_tokens: int = 4096,
    request_rate_per_sec: float = 10.0,
    has_rdma: bool = False,
    num_gpus: int = 1,
) -> dict:
    """Determine if Dynamo prefill/decode disaggregation helps this deployment."""
    variant, error = _resolve_supported_model(model)
    if error is not None:
        return error
    gpu_profile, error = _resolve_supported_gpu(gpu)
    if error is not None:
        return error

    traits = resolve_platform_traits(gpu_profile)
    long_prompts = avg_prompt_tokens >= 8192
    high_rate = request_rate_per_sec >= 8.0
    can_split = num_gpus >= 2
    has_fast_transport = has_rdma or traits.has_high_speed_interconnect

    recommended = bool(can_split and long_prompts and high_rate)
    rationale: list[str] = []
    warnings: list[str] = []
    if not can_split:
        rationale.append("Need at least 2 GPUs for aggregate/disaggregate serving.")
    elif not long_prompts:
        rationale.append("Average prompts are too short to justify prefill/decode separation.")
    elif not high_rate:
        rationale.append("Request rate is too low for disaggregation overhead to pay back.")
    else:
        rationale.append("Long prompts and sustained arrival rate favor decoupling prefill from decode.")

    if recommended and not has_fast_transport:
        warnings.append(
            "Split topology is still allowed, but RDMA or NVLink-class transport is "
            "strongly recommended for production reliability."
        )
    if gpu_profile.name == "H100 PCIe" and recommended:
        warnings.append(
            "H100 PCIe disaggregation is transport-sensitive; validate TTFT variance "
            "and cache handoff retries before production rollout."
        )

    return {
        "disaggregation": {
            "recommended": recommended,
            "topology": "prefill_decode_split" if recommended else "single_endpoint",
            "cache_backend": "lmcache",
            "lmcache_mode": "shared" if recommended else "local",
            "connector": "LMCache shared namespace",
            "required_observability_targets": (
                ["primary", "cache", "prefill", "decode"] if recommended else ["primary", "cache"]
            ),
            "rationale": rationale,
            "warnings": warnings,
            "configuration": {
                "prefill_gpus": max(1, num_gpus // 3) if recommended else 0,
                "decode_gpus": num_gpus - max(1, num_gpus // 3) if recommended else num_gpus,
                "session_affinity": True,
                "request_target": "primary",
            },
        },
        "model": variant.name,
        "gpu": gpu_profile.name,
        "target_profile": "dynamo_long_context_coding",
        "summary": (
            f"{'Recommended' if recommended else 'Not recommended'}: "
            f"Dynamo prefill/decode split for {variant.name} on {gpu_profile.name}."
        ),
        "confidence": 0.88,
        "evidence": "dynamo_disaggregation_policy",
    }


def compare_quantization(model: str, gpu: str) -> dict:
    """Compare the supported quantization options for the production lane."""
    variant, error = _resolve_supported_model(model)
    if error is not None:
        return error
    gpu_profile, error = _resolve_supported_gpu(gpu)
    if error is not None:
        return error

    options = [
        {
            "quantization": "fp8",
            "weight_gb": round(variant.weight_gb("fp8"), 1),
            "kv_cache_dtype": "fp8_e4m3",
            "accuracy": "default production quality/throughput tradeoff",
            "throughput_relative": "1.0x baseline for the supported lane",
            "supported": True,
            "recommended_rank": 1 if not (variant.name == "Kimi-K2.5" and gpu_profile.fp4_support) else 2,
            "notes": "Default for Hopper and also the fallback for Blackwell when FP4 headroom is not needed.",
        },
        {
            "quantization": "bf16",
            "weight_gb": round(variant.weight_gb("bf16"), 1),
            "kv_cache_dtype": "fp8_e4m3",
            "accuracy": "highest conservatism, lowest density",
            "throughput_relative": "0.6-0.7x vs FP8",
            "supported": True,
            "recommended_rank": 3,
            "notes": "Use for troubleshooting or quality triage, not the primary production lane.",
        },
    ]
    if gpu_profile.fp4_support and variant.name == "Kimi-K2.5":
        options.append(
            {
                "quantization": "fp4",
                "weight_gb": round(variant.weight_gb("fp4"), 1),
                "kv_cache_dtype": "fp8_e4m3",
                "accuracy": "needs production eval but is the density-maximizing Blackwell path",
                "throughput_relative": "1.3-1.6x vs FP8",
                "supported": True,
                "recommended_rank": 1,
                "notes": "Preferred on B200/B300 when Kimi-K2.5 needs more KV headroom or better session density.",
            }
        )

    options.sort(key=lambda item: item["recommended_rank"])

    return {
        "options": options,
        "model": variant.name,
        "gpu": gpu_profile.name,
        "target_profile": "dynamo_long_context_coding",
        "summary": f"Top pick for {variant.name} on {gpu_profile.name}: {options[0]['quantization']}.",
        "confidence": 0.9,
        "evidence": "target_quantization_policy",
    }
