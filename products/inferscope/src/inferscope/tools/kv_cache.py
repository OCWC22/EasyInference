"""KV cache management tools — tiering, offloading, disaggregation, budget calculation.

These tools are critical for long-context and agent workloads where KV cache
management is the primary bottleneck, not compute or scheduling.
"""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant, list_models
from inferscope.optimization.memory_planner import plan_memory
from inferscope.optimization.platform_policy import resolve_platform_traits
from inferscope.optimization.serving_profile import WorkloadMode


def calculate_kv_budget(
    model: str,
    context_length: int,
    batch_size: int = 1,
    kv_dtype: str = "fp8",
) -> dict:
    """Calculate exact KV cache memory requirement in bytes.

    Uses model architecture (layers, kv_heads, head_dim, MLA latent dim).
    Returns per-token, per-sequence, and total budget with tier recommendations.
    """
    variant = get_model_variant(model)
    if variant is None:
        return {
            "error": f"Unknown model: '{model}'",
            "available_models": list_models(),
            "confidence": 0.0,
        }

    # Per-token per-layer KV bytes
    kv_per_token_per_layer = variant.kv_cache_bytes_per_token(kv_dtype)
    kv_per_token_all_layers = kv_per_token_per_layer * variant.layers
    kv_per_sequence = kv_per_token_all_layers * context_length
    kv_total = kv_per_sequence * batch_size

    # Tier recommendations
    tiers = []
    kv_total_gb = kv_total / (1024**3)
    if kv_total_gb < 20:
        tiers.append({"tier": "G1_gpu_hbm", "reason": "Fits comfortably in GPU HBM", "recommended": True})
    elif kv_total_gb < 100:
        tiers.append({"tier": "G1_gpu_hbm", "reason": "Active decode sequences", "recommended": True})
        tiers.append({"tier": "G2_cpu_dram", "reason": "Overflow and paused sessions", "recommended": True})
    else:
        tiers.append({"tier": "G1_gpu_hbm", "reason": "Active decode sequences", "recommended": True})
        tiers.append(
            {
                "tier": "G2_cpu_dram",
                "reason": "Session persistence, prefix overflow",
                "recommended": True,
            }
        )
        tiers.append(
            {
                "tier": "G3_local_ssd",
                "reason": "Cold prefix cache, long-horizon agents",
                "recommended": True,
            }
        )

    is_mla = variant.attention_type == "MLA"

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
            "is_mla": is_mla,
            "mla_compression_ratio": variant.serving.get("compression_ratio", 1) if is_mla else 1,
        },
        "tier_recommendations": tiers,
        "model": variant.name,
        "summary": (
            f"{variant.name} @ {context_length // 1024}K context × {batch_size} batch ({kv_dtype}): "
            f"{kv_total_gb:.2f} GB KV cache"
            + (f" (MLA {variant.serving.get('compression_ratio', 32)}x compressed)" if is_mla else "")
        ),
        "confidence": 0.95,
        "evidence": "architecture_based_calculation",
    }


def recommend_kv_strategy(
    model: str,
    gpu: str,
    workload: str = "chat",
    max_context: int = 32768,
    concurrent_sessions: int = 100,
) -> dict:
    """Recommend KV cache tiering strategy for this deployment.

    Returns: which tiers to enable (GPU/CPU/SSD/remote), connector config,
    LMCache vs OffloadingConnector vs HiCache decision, and projected KV budget.
    """
    variant = get_model_variant(model)
    if variant is None:
        return {"error": f"Unknown model: '{model}'", "confidence": 0.0}

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None:
        return {"error": f"Unknown GPU: '{gpu}'", "confidence": 0.0}

    try:
        wm = WorkloadMode(workload)
    except ValueError:
        return {"error": f"Unknown workload: '{workload}'", "confidence": 0.0}

    is_mla = variant.attention_type == "MLA"
    traits = resolve_platform_traits(gpu_profile)

    # Calculate KV budget (with PagedAttention block fragmentation)
    kv_dtype = "fp8" if gpu_profile.fp8_support else "fp16"
    kv_per_token = variant.kv_cache_bytes_per_token(kv_dtype) * variant.layers
    kv_per_session = kv_per_token * max_context
    # PagedAttention block alignment wastes ~25% due to internal fragmentation
    # (partially-filled last blocks per sequence, block table overhead)
    fragmentation_overhead = 1.33  # 1/0.75 — need 33% more memory than theoretical
    total_kv_gb = (kv_per_session * concurrent_sessions * fragmentation_overhead) / (1024**3)

    # Memory plan for single-GPU slice
    mem = plan_memory(
        model=variant,
        gpu=gpu_profile,
        num_gpus=1,
        tp=1,
        precision="fp8" if gpu_profile.fp8_support else "bf16",
        kv_precision=kv_dtype,
    )

    strategy = {
        "tiers": [],
        "connector": "",
        "engine_recommendation": "",
        "notes": [],
    }

    gpu_kv_budget_gb = mem.kv_cache_budget_gb if mem.fits else 0

    # Tier selection logic
    if total_kv_gb <= gpu_kv_budget_gb * 0.8:
        strategy["tiers"] = ["G1_gpu_hbm"]
        strategy["notes"].append("All KV cache fits in GPU HBM — no offloading needed")
    elif total_kv_gb <= gpu_kv_budget_gb * 3:
        strategy["tiers"] = ["G1_gpu_hbm", "G2_cpu_dram"]
        strategy["notes"].append(
            f"KV cache ({total_kv_gb:.1f} GB) exceeds GPU budget ({gpu_kv_budget_gb:.1f} GB) — "
            "CPU DRAM offloading recommended"
        )
    else:
        strategy["tiers"] = ["G1_gpu_hbm", "G2_cpu_dram", "G3_local_ssd"]
        strategy["notes"].append(f"Large KV requirement ({total_kv_gb:.1f} GB) — multi-tier offloading recommended")

    if traits.is_grace:
        strategy["notes"].append(
            f"{gpu_profile.name} exposes Grace coherent overflow ({traits.grace_memory_gb:.0f} GB @ "
            f"{traits.grace_memory_bandwidth_gb_s:.0f} GB/s advisory) before falling back to conventional CPU offload."
        )

    # Connector recommendation
    if len(strategy["tiers"]) == 1:
        strategy["connector"] = "none (GPU-resident prefix caching is sufficient)"
        strategy["remote_backend"] = "none"
    elif gpu_profile.vendor == "nvidia":
        if wm in (WorkloadMode.CODING, WorkloadMode.AGENT):
            strategy["connector"] = "LMCacheConnectorV1 (content-addressed, cross-session sharing)"
            strategy["engine_recommendation"] = "SGLang with HiCache for RadixAttention + tiered cache"
        else:
            strategy["connector"] = "OffloadingConnector (simple GPU→CPU offload)"
            strategy["engine_recommendation"] = "vLLM with OffloadingConnector"
        strategy["remote_backend"] = "none"
        # Dynamo + NIXL recommendation for multi-GPU disaggregated setups
        if "G3_local_ssd" in strategy["tiers"] or total_kv_gb > gpu_kv_budget_gb * 2:
            strategy["notes"].append(
                "KV exceeds local tiers — consider Dynamo 1.0 for orchestrated disaggregated serving "
                "with NIXL KV transfer (RDMA/NVLink) to distribute KV pressure across prefill/decode pools."
            )
            strategy["orchestration_layer"] = "dynamo"
    elif gpu_profile.vendor == "amd":
        strategy["connector"] = "MooncakeConnector (RDMA-based, AMD-optimized)"
        strategy["remote_backend"] = "none"
        if is_mla:
            strategy["engine_recommendation"] = "ATOM standalone (MLA models on AMD)"
        else:
            strategy["engine_recommendation"] = "vLLM with AITER + CPU offloading"

    # MLA models need much less KV
    if is_mla:
        strategy["notes"].append(
            f"MLA attention compresses KV ~{variant.serving.get('compression_ratio', 32)}x — "
            "KV cache pressure is dramatically lower than dense-GQA models"
        )

    # Workload-specific advice
    if wm == WorkloadMode.AGENT:
        strategy["notes"].append(
            "Agent workload: session-sticky routing MANDATORY — "
            "standard LRU eviction will evict sessions before tool-call reuse"
        )
    if wm == WorkloadMode.CODING:
        strategy["notes"].append(
            "Coding workload: prefix canonicalization critical — "
            "remove timestamps, request IDs, and tool noise from cacheable prefixes"
        )

    # FP8 KV cache reminder
    if gpu_profile.fp8_support and kv_dtype == "fp8":
        strategy["notes"].append(
            "FP8 KV cache enabled — single highest-impact optimization for long context (2x memory savings)"
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
        "workload": workload,
        "max_context": max_context,
        "concurrent_sessions": concurrent_sessions,
        "summary": (
            f"{variant.name} on {gpu_profile.name} ({workload}): "
            f"{total_kv_gb:.1f} GB KV for {concurrent_sessions} sessions @ {max_context // 1024}K — "
            f"{'fits in GPU' if total_kv_gb <= gpu_kv_budget_gb else 'needs offloading'}"
        ),
        "confidence": 0.85,
        "evidence": "kv_budget_analysis",
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
    """Determine if prefill/decode disaggregation would help this deployment.

    Checks prompt length distribution, TTFT vs SLO, RDMA availability,
    and returns yes/no decision with connector recommendation.
    """
    variant = get_model_variant(model)
    if variant is None:
        return {"error": f"Unknown model: '{model}'", "confidence": 0.0}

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None:
        return {"error": f"Unknown GPU: '{gpu}'", "confidence": 0.0}

    decision = {
        "recommended": False,
        "rationale": [],
        "connector": "",
        "warnings": [],
        "configuration": {},
    }
    traits = resolve_platform_traits(gpu_profile)

    # Decision logic from spec
    short_prompts = avg_prompt_tokens < 4096
    low_rate = request_rate_per_sec < 10.0
    single_gpu = num_gpus <= 1

    if single_gpu:
        decision["rationale"].append("Single GPU — disaggregation not possible")
        decision["warnings"].append("Need at least 2 GPUs for prefill/decode separation")
    elif short_prompts:
        decision["rationale"].append(
            f"Average prompt ({avg_prompt_tokens} tokens) is short — "
            "disaggregation can DEGRADE performance 20-30% for short prompts"
        )
    elif not has_rdma and gpu_profile.nvlink_bandwidth_gb_s == 0 and gpu_profile.if_bandwidth_gb_s == 0:
        decision["rationale"].append(
            "No RDMA, NVLink, or Infinity Fabric — PCIe-only KV transfer kills latency benefit"
        )
        decision["warnings"].append(
            "CRITICAL: Do NOT deploy disaggregated serving without RDMA — "
            "documented severe bottleneck on TRT-LLM/Dynamo"
        )
    elif low_rate:
        decision["rationale"].append(
            f"Low request rate ({request_rate_per_sec}/sec) — "
            "disaggregation overhead outweighs benefit at low concurrency"
        )
    else:
        decision["recommended"] = True
        decision["rationale"].append(
            f"Long prompts ({avg_prompt_tokens} tokens) + high rate ({request_rate_per_sec}/sec) + "
            f"{'RDMA' if has_rdma else 'high-speed interconnect'} available — "
            "disaggregation will improve TTFT"
        )

        # Connector recommendation
        is_blackwell = traits.is_blackwell
        is_gb200 = traits.is_gb200
        has_nvlink5 = traits.has_nvlink5

        if gpu_profile.vendor == "nvidia":
            if has_rdma:
                decision["connector"] = "NixlConnector (UCX/libfabric/EFA)"
            elif gpu_profile.nvlink_bandwidth_gb_s > 0:
                decision["connector"] = "P2pNcclConnector (same-node NVLink)"
            else:
                decision["connector"] = "NixlConnector (UCX)"
        elif gpu_profile.vendor == "amd":
            decision["connector"] = "MooncakeConnector (RDMA-based, AMD-optimized)"

        # Blackwell ISA advantages for disaggregated serving
        if is_blackwell:
            blackwell_notes = []
            if has_nvlink5:
                blackwell_notes.append(
                    f"NVLink5 @ {gpu_profile.nvlink_bandwidth_gb_s:.0f} GB/s — "
                    "2x KV transfer bandwidth vs NVLink4 (Hopper), "
                    "enabling larger prefill chunks without decode starvation"
                )
            if gpu_profile.extra.get("decompression_engine"):
                blackwell_notes.append(
                    "nvCOMP decompression engine accelerates data I/O — "
                    "can reduce KV cache transfer volume between prefill/decode nodes"
                )
            if is_gb200:
                blackwell_notes.append(
                    "GB200 Grace Blackwell: prefill node can stage KV in Grace LPDDR5X "
                    "(480GB @ 546 GB/s) via NVLink-C2C before async transfer to decode node — "
                    "eliminates HBM pressure on prefill GPU during KV staging"
                )
            if gpu_profile.extra.get("helix_parallelism"):
                blackwell_notes.append(
                    "Helix parallelism: multi-node disaggregation across NVLink5 domains "
                    "with overlapped KV transfer and compute"
                )
            decision["blackwell_advantages"] = blackwell_notes

        # Suggested split
        prefill_gpus = max(1, num_gpus // 3)
        decode_gpus = num_gpus - prefill_gpus
        # Blackwell's faster KV transfer means decode can handle more load
        if is_blackwell and num_gpus >= 4:
            prefill_gpus = max(1, num_gpus // 4)
            decode_gpus = num_gpus - prefill_gpus
        decision["configuration"] = {
            "prefill_gpus": prefill_gpus,
            "decode_gpus": decode_gpus,
            "note": (
                f"Start with {prefill_gpus}P/{decode_gpus}D split, adjust based on TTFT/ITL metrics"
                + (". Blackwell NVLink5 enables higher decode:prefill ratio" if is_blackwell else "")
            ),
        }

    return {
        "disaggregation": decision,
        "model": variant.name,
        "gpu": gpu_profile.name,
        "summary": (
            f"{'✅ Recommended' if decision['recommended'] else '❌ Not recommended'}: "
            f"P/D disaggregation for {variant.name} on {gpu_profile.name} — "
            + (decision["rationale"][0] if decision["rationale"] else "")
        ),
        "confidence": 0.80 if decision["recommended"] else 0.85,
        "evidence": "disaggregation_decision_rules",
    }


def compare_quantization(model: str, gpu: str) -> dict:
    """Compare quantization options with accuracy, memory, and throughput tradeoffs.

    Returns all viable options for this specific GPU, ranked by recommendation.
    """
    variant = get_model_variant(model)
    if variant is None:
        return {"error": f"Unknown model: '{model}'", "confidence": 0.0}

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None:
        return {"error": f"Unknown GPU: '{gpu}'", "confidence": 0.0}

    options = []

    # BF16/FP16 — always available
    options.append(
        {
            "quantization": "bf16",
            "weight_gb": round(variant.weight_gb("bf16"), 1),
            "kv_cache_dtype": "fp16",
            "accuracy": "baseline (no loss)",
            "throughput_relative": "1.0x",
            "supported": True,
            "recommended_rank": 3,
            "notes": "Full precision — use as accuracy baseline",
        }
    )

    # FP8
    if gpu_profile.fp8_support:
        options.append(
            {
                "quantization": "fp8",
                "weight_gb": round(variant.weight_gb("fp8"), 1),
                "kv_cache_dtype": "fp8_e4m3",
                "accuracy": "negligible loss (<0.5% on most benchmarks)",
                "throughput_relative": "1.8-2.0x vs BF16",
                "supported": True,
                "recommended_rank": 1,
                "notes": (
                    f"Native FP8 on {gpu_profile.name} ({gpu_profile.fp8_format}). "
                    "Best general-purpose option — halves memory, near-full accuracy."
                    + (" FNUZ format — auto-converted from OCP by vLLM." if gpu_profile.fp8_format == "FNUZ" else "")
                ),
            }
        )
    elif gpu_profile.architecture == "Ampere":
        options.append(
            {
                "quantization": "fp8 (W8A16 Marlin)",
                "weight_gb": round(variant.weight_gb("fp8"), 1),
                "kv_cache_dtype": "fp16",
                "accuracy": "weight-only dequant — slightly less accurate than native FP8",
                "throughput_relative": "1.3-1.5x vs BF16",
                "supported": True,
                "recommended_rank": 2,
                "notes": "Ampere FP8 via Marlin — weight-only, no native FP8 compute",
            }
        )

    # FP4 variants
    if gpu_profile.fp4_support:
        fmt = gpu_profile.fp4_format
        options.append(
            {
                "quantization": fmt.lower(),
                "weight_gb": round(variant.weight_gb("fp4"), 1),
                "kv_cache_dtype": "fp8_e4m3" if gpu_profile.fp8_support else "fp16",
                "accuracy": "<1% loss on most tasks, EVAL CODING/TOOL quality first",
                "throughput_relative": "2.5-3.5x vs BF16",
                "supported": True,
                "recommended_rank": 2,
                "notes": (
                    f"Native {fmt} on {gpu_profile.name}. "
                    f"{'Scale granularity: 16 elements' if fmt == 'NVFP4' else 'Scale granularity: 32 elements'}. "
                    "MUST evaluate coding/tool-use accuracy before production."
                ),
            }
        )

    # INT8
    options.append(
        {
            "quantization": "int8",
            "weight_gb": round(variant.weight_gb("int8"), 1),
            "kv_cache_dtype": "fp16",
            "accuracy": "minor loss, well-studied",
            "throughput_relative": "1.5-1.8x vs BF16",
            "supported": True,
            "recommended_rank": 2 if not gpu_profile.fp8_support else 4,
            "notes": "Good fallback when FP8 not available. Well-supported on all GPUs.",
        }
    )

    # AWQ/GPTQ (INT4 weight-only)
    options.append(
        {
            "quantization": "awq",
            "weight_gb": round(variant.weight_gb("int4"), 1),
            "kv_cache_dtype": "fp16",
            "accuracy": "1-3% loss, calibration-dependent",
            "throughput_relative": "1.5-2.0x vs BF16 (memory-bound scenarios)",
            "supported": True,
            "recommended_rank": 2 if gpu_profile.architecture == "Ampere" else 5,
            "notes": (
                "4-bit weight-only quantization. Best for Ampere (no FP8). "
                + (
                    "Uses Marlin kernels on NVIDIA."
                    if gpu_profile.vendor == "nvidia"
                    else "Uses Triton backends on AMD (slower than NVIDIA Marlin)."
                )
            ),
        }
    )

    # Sort by recommended rank
    options.sort(key=lambda x: x["recommended_rank"])

    return {
        "options": options,
        "model": variant.name,
        "gpu": gpu_profile.name,
        "summary": (
            f"Top pick for {variant.name} on {gpu_profile.name}: "
            f"{options[0]['quantization']} ({options[0]['weight_gb']} GB, {options[0]['throughput_relative']})"
        ),
        "confidence": 0.85,
        "evidence": "hardware_format_compatibility",
    }
