"""Pre-flight validation — checks if a serving config will work before deployment.

Validates TP divisibility, memory fit, format compatibility, known bugs, etc.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.models.registry import ModelVariant
from inferscope.optimization.memory_planner import plan_memory
from inferscope.optimization.platform_policy import (
    EngineSupportTier,
    resolve_engine_support,
    resolve_preferred_tp,
)
from inferscope.optimization.serving_profile import WorkloadMode


@dataclass
class ValidationResult:
    """Result of pre-flight validation."""

    valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "valid": self.valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "info": self.info,
        }


def validate_config(
    model: ModelVariant,
    gpu: GPUProfile,
    tp: int = 1,
    quantization: str = "auto",
    engine: str = "vllm",
) -> ValidationResult:
    """Validate a serving configuration before deployment.

    Checks:
    - TP divides num_attention_heads and num_kv_heads
    - Model fits in GPU memory
    - Quantization format is supported by GPU
    - Known engine/model/GPU incompatibilities
    """
    result = ValidationResult()

    tp_min = model.serving.get("tp_min")
    if isinstance(tp_min, int) and tp < tp_min:
        result.valid = False
        result.errors.append(f"TP={tp} is below the model minimum tp_min={tp_min}.")

    # --- TP divisibility ---
    if model.kv_heads > 0 and tp > 1 and model.kv_heads % tp != 0:
        result.valid = False
        result.errors.append(
            f"TP={tp} does not evenly divide num_kv_heads={model.kv_heads}. "
            f"Valid TP values: {[i for i in range(1, model.kv_heads + 1) if model.kv_heads % i == 0]}"
        )

    # --- Quantization compatibility ---
    if quantization == "auto":
        quantization = "fp8" if gpu.fp8_support else "bf16"
    normalized_precision = "fp4" if quantization in ("nvfp4", "mxfp4") else quantization

    if normalized_precision in ("fp8", "fp8_e4m3") and not gpu.fp8_support:
        # Ampere supports FP8 via W8A16 Marlin (weight-only dequant) — not native, but functional
        if gpu.architecture == "Ampere":
            result.warnings.append(
                f"FP8 on {gpu.name} ({gpu.architecture}) uses W8A16 Marlin weight-only dequant, "
                f"NOT native FP8 compute. Performance is lower than Hopper+/CDNA3+ native FP8. "
                f"Consider INT8/AWQ/GPTQ for potentially better Ampere performance."
            )
        else:
            result.valid = False
            result.errors.append(
                f"FP8 quantization requires Hopper+ or CDNA3+. "
                f"{gpu.name} ({gpu.architecture}) does not support native FP8. "
                f"Use INT8/AWQ/GPTQ instead."
            )

    if quantization in ("nvfp4", "fp4", "mxfp4") and not gpu.fp4_support:
        result.valid = False
        result.errors.append(
            f"FP4 quantization requires Blackwell (NVFP4) or CDNA4 (MXFP4). "
            f"{gpu.name} ({gpu.architecture}) does not support FP4."
        )

    if quantization == "nvfp4" and gpu.fp4_format == "MXFP4":
        result.valid = False
        result.errors.append(
            f"NVFP4 is NVIDIA Blackwell only. {gpu.name} supports MXFP4, not NVFP4. Use MXFP4 quantization instead."
        )

    if quantization == "mxfp4" and gpu.fp4_format == "NVFP4":
        result.valid = False
        result.errors.append(
            f"MXFP4 is AMD CDNA4 native. {gpu.name} supports NVFP4, not MXFP4. Use NVFP4 quantization instead."
        )

    # --- FP8 format compatibility ---
    if gpu.fp8_support and gpu.fp8_format == "FNUZ":
        result.warnings.append(
            f"{gpu.name} uses FNUZ FP8 format (not OCP). "
            f"Models trained with OCP FP8 will be auto-converted by vLLM with 2x scale factor."
        )

    # --- Memory fit ---
    precision = normalized_precision if normalized_precision not in ("auto",) else "fp16"
    mem = plan_memory(model=model, gpu=gpu, num_gpus=tp, tp=tp, precision=precision)
    if not mem.fits:
        result.valid = False
        result.errors.append(
            f"Model does not fit: weights need {mem.weight_gb:.1f} GB/GPU, "
            f"but only {mem.usable_memory_gb / tp:.1f} GB usable per GPU "
            f"(after {gpu.memory_gb} GB × 0.92 utilization / TP={tp})."
        )
    else:
        result.info.append(
            f"Memory: {mem.weight_gb:.1f} GB weights/GPU, "
            f"{mem.kv_cache_budget_gb:.1f} GB KV cache budget, "
            f"~{mem.max_concurrent_sequences} max concurrent sequences"
        )
        if mem.platform_overflow_tier != "gpu_only":
            result.info.append(
                f"Platform overflow advisory: {mem.platform_overflow_tier} (+{mem.overflow_memory_gb:.0f} GB)."
            )

    # --- Engine-specific checks ---
    support = resolve_engine_support(engine, gpu, multi_node=tp > 1)
    if support.tier == EngineSupportTier.UNSUPPORTED:
        result.valid = False
        result.errors.append(support.reason)
    elif support.tier == EngineSupportTier.PREVIEW:
        result.warnings.append(f"Preview engine: {support.reason}")

    if engine == "atom" and gpu.vendor != "amd":
        result.valid = False
        result.errors.append("ATOM engine only works on AMD GPUs (MI300X/MI325X/MI355X)")

    # DeepSeek MLA on ROCm needs block-size 1
    if model.attention_type == "MLA" and gpu.vendor == "amd" and engine in ("vllm",):
        result.warnings.append("DeepSeek MLA models require --block-size 1 on ROCm for correct results")

    # AITER mandatory on AMD
    if gpu.vendor == "amd":
        result.info.append("AMD GPU detected — VLLM_ROCM_USE_AITER=1 is mandatory for competitive performance")
        if gpu.compute_capability == "gfx942":
            result.warnings.append("MI300X (gfx942): VLLM_ROCM_USE_AITER_FP8BMM MUST be 0 — crashes with memory faults")

    # MoE without EP warning
    if model.model_type == "moe" and model.experts_total > 64 and tp > 1:
        result.warnings.append(
            f"Large MoE model ({model.experts_total} experts) — "
            f"consider Expert Parallelism (EP) in addition to TP for better scaling"
        )

    # Ampere FP8 misconception
    if gpu.architecture == "Ampere" and quantization == "fp8":
        result.info.append("FP8 on Ampere uses W8A16 Marlin (weight-only dequant), not native FP8 compute")

    preferred_tp, preferred_reason = resolve_preferred_tp(
        model,
        gpu,
        num_gpus=max(tp, 1),
        precision=normalized_precision,
        workload=WorkloadMode.CHAT,
    )
    if preferred_tp is not None and preferred_tp != tp and preferred_reason:
        result.warnings.append(f"Platform/model hint prefers TP={preferred_tp}. {preferred_reason}")

    return result
