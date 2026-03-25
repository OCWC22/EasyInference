"""Recommendation tools — config generation, engine selection, parallelism."""

from __future__ import annotations

from inferscope.hardware.gpu_profiles import get_gpu_profile, list_gpus
from inferscope.models.registry import get_model_variant, list_models
from inferscope.optimization.platform_policy import EngineSupportTier, resolve_engine_support
from inferscope.optimization.recommender import recommend
from inferscope.optimization.serving_profile import EngineType, WorkloadMode
from inferscope.security import InputValidationError, validate_gpu_name, validate_model_name, validate_positive_int


def _safe_lookup(model: str, gpu: str) -> tuple:
    """Validate and look up model + GPU. Returns (variant, gpu_profile) or (None, error_dict)."""
    try:
        model = validate_model_name(model)
    except InputValidationError as e:
        return None, {"error": str(e), "confidence": 0.0}
    try:
        gpu = validate_gpu_name(gpu)
    except InputValidationError as e:
        return None, {"error": str(e), "confidence": 0.0}

    variant = get_model_variant(model)
    if variant is None:
        return None, {"error": f"Unknown model: '{model}'", "available_models": list_models(), "confidence": 0.0}

    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None:
        return None, {"error": f"Unknown GPU: '{gpu}'", "available_gpus": list_gpus(), "confidence": 0.0}

    return (variant, gpu_profile), None


def recommend_config(
    model: str,
    gpu: str,
    workload: str = "chat",
    num_gpus: int = 1,
    engine: str = "auto",
) -> dict:
    """Generate optimal serving config for a model+GPU+workload combination.

    Returns a normalized ServingProfile AND the engine-specific compiled flags.
    Engine auto-selection: ATOM for MLA/MoE on AMD, vLLM for NVIDIA default,
    SGLang for cache-heavy, TRT-LLM for max NVIDIA throughput.
    """
    result = _safe_lookup(model, gpu)
    if result[1] is not None:
        return result[1]
    variant, gpu_profile = result[0]

    try:
        wm = WorkloadMode(workload)
    except ValueError:
        return {
            "error": f"Unknown workload: '{workload}'. Use: coding, chat, agent, long_context_rag",
            "confidence": 0.0,
        }

    try:
        num_gpus = validate_positive_int(num_gpus, "num_gpus", max_value=1024)
    except InputValidationError as e:
        return {"error": str(e), "confidence": 0.0}

    try:
        profile, engine_config, mem_plan = recommend(
            model=variant,
            gpu=gpu_profile,
            num_gpus=num_gpus,
            workload=wm,
            engine=engine,
        )
    except ValueError as e:
        return {"error": str(e), "confidence": 0.0}

    return {
        "serving_profile": profile.to_dict(),
        "engine_config": engine_config.to_dict(),
        "memory_plan": mem_plan.to_dict(),
        "summary": (
            f"Recommended: {variant.name} on {num_gpus}× {gpu_profile.name} | "
            f"Engine: {engine_config.engine} | "
            f"TP={profile.topology.tp} DP={profile.topology.dp} EP={profile.topology.ep} | "
            f"Precision: {profile.precision.weights} | "
            f"Workload: {workload} | "
            f"Support: {engine_config.support_tier} | "
            f"{'✅ fits' if mem_plan.fits else '❌ does not fit'}"
        ),
        "launch_command": engine_config.command,
        "confidence": 0.85 if mem_plan.fits else 0.6,
        "evidence": "knowledge_based_recommendation",
    }


def recommend_engine(
    model: str,
    gpu: str,
    workload: str = "chat",
    num_gpus: int = 1,
    multi_node: bool = False,
) -> dict:
    """Recommend the best inference engine for this model+GPU+workload.

    Returns ranked options with rationale.
    """
    result = _safe_lookup(model, gpu)
    if result[1] is not None:
        return result[1]
    variant, gpu_profile = result[0]

    try:
        wm = WorkloadMode(workload)
    except ValueError:
        return {"error": f"Unknown workload: '{workload}'", "confidence": 0.0}

    # Generate recommendations for all viable engines
    _, selected_engine_config, _ = recommend(
        model=variant,
        gpu=gpu_profile,
        num_gpus=num_gpus,
        workload=wm,
        engine="auto",
    )
    selected_engine = selected_engine_config.engine
    tier_priority = {
        EngineSupportTier.RECOMMENDED: 1,
        EngineSupportTier.SUPPORTED: 2,
        EngineSupportTier.PREVIEW: 3,
    }
    rankings = []
    is_multi_node = multi_node or num_gpus > 1
    for candidate in EngineType:
        support = resolve_engine_support(candidate, gpu_profile, multi_node=is_multi_node)
        if support.tier == EngineSupportTier.UNSUPPORTED:
            continue

        rank = tier_priority[support.tier] * 10
        best_for = "General inference planning"
        rationale = support.reason

        if candidate.value == selected_engine:
            rank = 0
            rationale = f"Matches InferScope's full DAG recommendation. {support.reason}"
        elif candidate == EngineType.SGLANG and wm in (WorkloadMode.CODING, WorkloadMode.AGENT):
            rank -= 3
            best_for = "Coding copilots, multi-turn agents, and prefix-heavy workloads"
            rationale = f"Prefix-heavy workload bias. {support.reason}"
        elif candidate == EngineType.VLLM:
            best_for = "Broad compatibility and first-class InferScope support"
        elif candidate == EngineType.TRTLLM:
            best_for = "Manual Blackwell throughput experiments after explicit validation"
        elif candidate == EngineType.DYNAMO:
            best_for = "Multi-node or disaggregated planning experiments"
        elif candidate == EngineType.ATOM:
            best_for = "AMD MLA/MoE deployments"

        rankings.append(
            {
                "engine": candidate.value,
                "rank": rank,
                "rationale": rationale,
                "best_for": best_for,
                "support_tier": support.tier.value,
                "support_reason": support.reason,
            }
        )

    rankings.sort(key=lambda x: x["rank"])

    return {
        "rankings": rankings,
        "model": variant.name,
        "gpu": gpu_profile.name,
        "workload": workload,
        "selected_engine": selected_engine,
        "summary": f"Top pick: {rankings[0]['engine']} — {rankings[0]['rationale']}",
        "confidence": 0.85,
        "evidence": "engine_selection_rules",
    }


def suggest_parallelism(model: str, gpu: str, num_gpus: int) -> dict:
    """Recommend TP/PP/DP/EP strategy for model+GPU combination."""
    result = _safe_lookup(model, gpu)
    if result[1] is not None:
        return result[1]
    variant, gpu_profile = result[0]

    try:
        num_gpus = validate_positive_int(num_gpus, "num_gpus", max_value=1024)
    except InputValidationError as e:
        return {"error": str(e), "confidence": 0.0}

    # Calculate for multiple precision options
    suggestions = []
    for prec in ["fp8", "bf16"]:
        if prec == "fp8" and not gpu_profile.fp8_support:
            continue

        weight_gb = variant.weight_gb(prec)
        per_gpu_usable = gpu_profile.memory_gb * 0.90

        tp = 1
        while weight_gb / tp > per_gpu_usable * 0.75:
            tp *= 2
            if tp > num_gpus:
                break
        tp = min(tp, num_gpus)

        ep = 1
        if variant.model_type == "moe" and variant.experts_total > 64 and num_gpus >= 4:
            ep = min(2, num_gpus // tp)

        dp = max(1, num_gpus // (tp * ep))

        suggestions.append(
            {
                "precision": prec,
                "tp": tp,
                "pp": 1,
                "dp": dp,
                "ep": ep,
                "weight_gb_per_gpu": round(weight_gb / tp, 1),
                "kv_headroom_gb_per_gpu": round(per_gpu_usable - weight_gb / tp, 1),
                "notes": [],
            }
        )

        # Add notes
        if tp > 1 and (gpu_profile.nvlink_bandwidth_gb_s == 0 and gpu_profile.if_bandwidth_gb_s == 0):
            suggestions[-1]["notes"].append(
                f"WARNING: TP={tp} without NVLink/IF — PCIe-only TP has high latency overhead"
            )
        if ep > 1:
            suggestions[-1]["notes"].append(f"Expert Parallelism EP={ep} for {variant.experts_total}-expert MoE")

    return {
        "suggestions": suggestions,
        "model": variant.name,
        "gpu": gpu_profile.name,
        "num_gpus": num_gpus,
        "summary": (
            f"{variant.name} on {num_gpus}× {gpu_profile.name}: "
            + ", ".join(f"{s['precision'].upper()}: TP={s['tp']} DP={s['dp']} EP={s['ep']}" for s in suggestions)
        ),
        "confidence": 0.85,
        "evidence": "memory_topology_analysis",
    }
