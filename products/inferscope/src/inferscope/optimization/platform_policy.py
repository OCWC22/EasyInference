"""Shared platform policy for engine selection and Hopper/Blackwell tuning."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.models.registry import ModelVariant
from inferscope.optimization.serving_profile import EngineType, PrecisionSpec, WorkloadMode


class PlatformFamily(StrEnum):
    """Normalized hardware platform families used across the optimizer."""

    OTHER = "other"
    AMPERE = "ampere"
    HOPPER = "hopper"
    HOPPER_PCIE = "hopper_pcie"
    HOPPER_GRACE = "hopper_grace"
    BLACKWELL = "blackwell"
    BLACKWELL_ULTRA = "blackwell_ultra"
    BLACKWELL_GRACE = "blackwell_grace"
    BLACKWELL_ULTRA_GRACE = "blackwell_ultra_grace"
    CDNA3 = "cdna3"
    CDNA4 = "cdna4"


class EngineSupportTier(StrEnum):
    """Public support tier for an engine on a platform."""

    RECOMMENDED = "recommended"
    SUPPORTED = "supported"
    PREVIEW = "preview"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class PlatformTraits:
    """Resolved platform capabilities derived from a GPU profile."""

    family: PlatformFamily
    gpu_type: str
    vendor: str
    architecture: str
    is_nvidia: bool
    is_amd: bool
    is_ampere: bool
    is_hopper: bool
    is_hopper_pcie: bool
    is_h200: bool
    is_blackwell: bool
    is_b300: bool
    is_grace: bool
    is_gh200: bool
    is_gb200: bool
    is_gb300: bool
    has_high_speed_interconnect: bool
    has_nvlink5: bool
    has_decompression_engine: bool
    has_helix_parallelism: bool
    has_accelerated_softmax: bool
    grace_memory_gb: float
    grace_memory_bandwidth_gb_s: float
    c2c_bandwidth_gb_s: float
    overflow_tier: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family.value,
            "gpu_type": self.gpu_type,
            "vendor": self.vendor,
            "architecture": self.architecture,
            "is_ampere": self.is_ampere,
            "is_hopper": self.is_hopper,
            "is_hopper_pcie": self.is_hopper_pcie,
            "is_h200": self.is_h200,
            "is_blackwell": self.is_blackwell,
            "is_b300": self.is_b300,
            "is_grace": self.is_grace,
            "is_gh200": self.is_gh200,
            "is_gb200": self.is_gb200,
            "is_gb300": self.is_gb300,
            "has_high_speed_interconnect": self.has_high_speed_interconnect,
            "has_nvlink5": self.has_nvlink5,
            "has_decompression_engine": self.has_decompression_engine,
            "has_helix_parallelism": self.has_helix_parallelism,
            "has_accelerated_softmax": self.has_accelerated_softmax,
            "grace_memory_gb": self.grace_memory_gb,
            "grace_memory_bandwidth_gb_s": self.grace_memory_bandwidth_gb_s,
            "c2c_bandwidth_gb_s": self.c2c_bandwidth_gb_s,
            "overflow_tier": self.overflow_tier,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class EngineSupport:
    """Support status for a specific engine on a specific platform."""

    engine: str
    tier: EngineSupportTier
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "engine": self.engine,
            "support_tier": self.tier.value,
            "support_reason": self.reason,
        }


def resolve_platform_traits(gpu: GPUProfile) -> PlatformTraits:
    """Resolve normalized platform traits from a GPU profile."""

    extra = gpu.extra or {}
    is_nvidia = gpu.vendor == "nvidia"
    is_amd = gpu.vendor == "amd"
    is_ampere = gpu.architecture == "Ampere"
    is_hopper = gpu.architecture == "Hopper"
    is_blackwell = gpu.architecture == "Blackwell"
    is_hopper_pcie = is_hopper and gpu.compute_capability == "sm_90"
    is_h200 = is_hopper and gpu.memory_gb >= 140
    is_grace = extra.get("grace_cpu_cores", 0) > 0
    is_gh200 = is_hopper and is_grace
    is_gb200 = is_blackwell and is_grace and gpu.compute_capability == "sm_100"
    is_gb300 = is_blackwell and is_grace and gpu.compute_capability == "sm_103"
    is_b300 = is_blackwell and gpu.compute_capability == "sm_103" and not is_grace
    has_high_speed_interconnect = bool(gpu.nvlink_bandwidth_gb_s or gpu.if_bandwidth_gb_s)
    has_nvlink5 = gpu.nvlink_version >= 5

    if is_ampere:
        family = PlatformFamily.AMPERE
    elif is_hopper and is_grace:
        family = PlatformFamily.HOPPER_GRACE
    elif is_hopper_pcie:
        family = PlatformFamily.HOPPER_PCIE
    elif is_hopper:
        family = PlatformFamily.HOPPER
    elif is_gb300:
        family = PlatformFamily.BLACKWELL_ULTRA_GRACE
    elif is_b300:
        family = PlatformFamily.BLACKWELL_ULTRA
    elif is_gb200:
        family = PlatformFamily.BLACKWELL_GRACE
    elif is_blackwell:
        family = PlatformFamily.BLACKWELL
    elif gpu.compute_capability == "gfx942":
        family = PlatformFamily.CDNA3
    elif gpu.compute_capability == "gfx950":
        family = PlatformFamily.CDNA4
    else:
        family = PlatformFamily.OTHER

    notes: list[str] = []
    if is_gh200:
        notes.append("Grace Hopper coherent overflow is available for KV spill and staging.")
    if is_gb200:
        notes.append("GB200 exposes Grace LPDDR5X over NVLink-C2C for coherent overflow.")
    if is_gb300:
        notes.append("GB300 combines Blackwell Ultra compute with Grace coherent overflow.")
    if is_b300:
        notes.append("B300 adds accelerated softmax and larger HBM headroom without Grace.")

    return PlatformTraits(
        family=family,
        gpu_type=gpu.name,
        vendor=gpu.vendor,
        architecture=gpu.architecture,
        is_nvidia=is_nvidia,
        is_amd=is_amd,
        is_ampere=is_ampere,
        is_hopper=is_hopper,
        is_hopper_pcie=is_hopper_pcie,
        is_h200=is_h200,
        is_blackwell=is_blackwell,
        is_b300=is_b300,
        is_grace=is_grace,
        is_gh200=is_gh200,
        is_gb200=is_gb200,
        is_gb300=is_gb300,
        has_high_speed_interconnect=has_high_speed_interconnect,
        has_nvlink5=has_nvlink5,
        has_decompression_engine=bool(extra.get("decompression_engine")),
        has_helix_parallelism=bool(extra.get("helix_parallelism")),
        has_accelerated_softmax=bool(extra.get("accelerated_softmax")),
        grace_memory_gb=float(extra.get("grace_memory_gb", 0.0) or 0.0),
        grace_memory_bandwidth_gb_s=float(extra.get("grace_memory_bandwidth_gb_s", 0.0) or 0.0),
        c2c_bandwidth_gb_s=float(extra.get("nvlink_c2c_bandwidth_gb_s", 0.0) or 0.0),
        overflow_tier="gpu_grace_coherent" if is_grace else "gpu_only",
        notes=notes,
    )


def resolve_engine_support(engine: EngineType | str, gpu: GPUProfile, multi_node: bool = False) -> EngineSupport:
    """Return support tier metadata for an engine on a platform."""

    engine_name = engine.value if isinstance(engine, EngineType) else str(engine).lower().strip()
    traits = resolve_platform_traits(gpu)

    if engine_name == "atom":
        if traits.is_amd:
            return EngineSupport(
                engine="atom",
                tier=EngineSupportTier.SUPPORTED,
                reason="ATOM is available for AMD deployments, especially MLA/MoE-heavy workloads.",
            )
        return EngineSupport(
            engine="atom",
            tier=EngineSupportTier.UNSUPPORTED,
            reason="ATOM is AMD-only and is not available on NVIDIA Hopper or Blackwell.",
        )

    if engine_name == "vllm":
        if traits.is_nvidia:
            return EngineSupport(
                engine="vllm",
                tier=EngineSupportTier.RECOMMENDED,
                reason="vLLM is the primary supported NVIDIA engine in InferScope for Hopper and Blackwell.",
            )
        if traits.is_amd:
            return EngineSupport(
                engine="vllm",
                tier=EngineSupportTier.SUPPORTED,
                reason="vLLM is the broad-compatibility path for AMD deployments in InferScope.",
            )

    if engine_name == "sglang":
        if traits.is_nvidia:
            return EngineSupport(
                engine="sglang",
                tier=EngineSupportTier.SUPPORTED,
                reason=(
                    "SGLang is a supported alternative for NVIDIA, strongest on prefix-heavy coding and agent flows."
                ),
            )
        if traits.is_amd:
            return EngineSupport(
                engine="sglang",
                tier=EngineSupportTier.SUPPORTED,
                reason="SGLang is usable on AMD, but InferScope still favors vLLM or ATOM for most deployments.",
            )

    if engine_name == "trtllm":
        if not traits.is_nvidia:
            return EngineSupport(
                engine="trtllm",
                tier=EngineSupportTier.UNSUPPORTED,
                reason="TensorRT-LLM is NVIDIA-only.",
            )
        return EngineSupport(
            engine="trtllm",
            tier=EngineSupportTier.SUPPORTED,
            reason=(
                "TensorRT-LLM v1.2+ is a supported NVIDIA engine with the highest compiled throughput; "
                "requires a compilation step and NVIDIA-only hardware."
            ),
        )

    if engine_name == "dynamo":
        if not traits.is_nvidia:
            return EngineSupport(
                engine="dynamo",
                tier=EngineSupportTier.UNSUPPORTED,
                reason="NVIDIA Dynamo is NVIDIA-only.",
            )
        if multi_node:
            return EngineSupport(
                engine="dynamo",
                tier=EngineSupportTier.RECOMMENDED,
                reason=(
                    "Dynamo 1.0 is the recommended orchestration layer for multi-node NVIDIA "
                    "disaggregated serving with KV-aware routing, NIXL transfer, and SLO-driven autoscaling."
                ),
            )
        return EngineSupport(
            engine="dynamo",
            tier=EngineSupportTier.SUPPORTED,
            reason=(
                "Dynamo 1.0 is a production orchestration layer for NVIDIA deployments. "
                "Recommended for multi-node/disaggregated topologies; supported for single-node."
            ),
        )

    return EngineSupport(
        engine=engine_name,
        tier=EngineSupportTier.UNSUPPORTED,
        reason=f"Unknown engine '{engine_name}'.",
    )


def resolve_preferred_precision(
    model: ModelVariant,
    gpu: GPUProfile,
    workload: WorkloadMode,
    num_gpus: int = 1,
) -> tuple[PrecisionSpec, str]:
    """Return the preferred precision policy for a model/GPU/workload."""

    traits = resolve_platform_traits(gpu)
    kv_cache = "fp8_e4m3" if gpu.fp8_support else "auto"

    if model.params_total_b <= 13 and gpu.memory_gb >= 40:
        return (
            PrecisionSpec(weights="bf16", activations="bf16", kv_cache=kv_cache),
            "Small model fits comfortably; prefer BF16 for minimal conversion risk.",
        )

    if (
        traits.is_blackwell
        and gpu.fp4_support
        and workload not in (WorkloadMode.CODING, WorkloadMode.AGENT)
        and (model.serving.get("nvidia_nvfp4") or model.serving.get("nvidia_fp4"))
    ):
        return (
            PrecisionSpec(weights="fp4", activations="fp8" if gpu.fp8_support else "bf16", kv_cache=kv_cache),
            "Blackwell-specific NVFP4 path is available for this model; prefer FP4 for throughput and memory headroom.",
        )

    fp8_fits_gpu_count = model.weight_gb("fp8") / max(num_gpus, 1) <= gpu.memory_gb * 0.92

    if gpu.fp8_support:
        if not fp8_fits_gpu_count and model.params_total_b > 30 and not traits.is_blackwell:
            return (
                PrecisionSpec(weights="awq", activations="fp16", kv_cache="auto"),
                "Native FP8 would overrun per-GPU HBM at the requested GPU count; "
                "falling back to AWQ for a memory-valid auto plan.",
            )
        if traits.is_amd and model.model_type == "moe":
            return (
                PrecisionSpec(weights="bf16", activations="bf16", kv_cache="fp8_e4m3"),
                "AMD MoE fallback: keep weights in BF16 and only compress KV to FP8 to avoid known decode regressions.",
            )
        return (
            PrecisionSpec(weights="fp8", activations="fp8", kv_cache=kv_cache),
            "Native FP8 support detected; use FP8 as the default throughput/quality tradeoff.",
        )

    if model.params_total_b > 30:
        return (
            PrecisionSpec(weights="awq", activations="fp16", kv_cache="auto"),
            "No native FP8 support and model is large; prefer AWQ/INT4-style weight compression.",
        )

    return (
        PrecisionSpec(weights="bf16", activations="bf16", kv_cache="auto"),
        "Defaulting to BF16 because the platform lacks native FP8.",
    )


def resolve_preferred_tp(
    model: ModelVariant,
    gpu: GPUProfile,
    num_gpus: int,
    precision: str,
    workload: WorkloadMode,
) -> tuple[int | None, str | None]:
    """Resolve TP from model serving hints before falling back to memory heuristics."""

    del workload  # reserved for future workload-specific TP overrides

    valid_tps = _valid_tps(model, num_gpus)
    if not valid_tps:
        return None, None

    traits = resolve_platform_traits(gpu)
    normalized_precision = precision.lower()

    hints: list[tuple[int, str]] = []
    if normalized_precision == "fp8":
        if traits.is_h200 and isinstance(model.serving.get("tp_fp8_h200"), int):
            hints.append((model.serving["tp_fp8_h200"], "Using model-specific H200 FP8 tensor-parallel hint."))
        elif traits.is_hopper and isinstance(model.serving.get("tp_fp8_h100"), int):
            hints.append((model.serving["tp_fp8_h100"], "Using model-specific H100 FP8 tensor-parallel hint."))
        elif isinstance(model.serving.get("tp_fp8"), int):
            hints.append((model.serving["tp_fp8"], "Using model-specific FP8 tensor-parallel hint."))
    elif normalized_precision in {"bf16", "fp16"}:
        if traits.family in (PlatformFamily.CDNA3, PlatformFamily.CDNA4):
            amd_hint = (
                model.serving.get("tp_fp16_mi300x") or model.serving.get("tp_bf16") or model.serving.get("tp_fp16")
            )
            if isinstance(amd_hint, int):
                hints.append((amd_hint, "Using AMD full-precision tensor-parallel hint."))
        else:
            fp_hint = model.serving.get("tp_bf16") or model.serving.get("tp_fp16")
            if isinstance(fp_hint, int):
                hints.append((fp_hint, "Using model-specific BF16/FP16 tensor-parallel hint."))
    elif normalized_precision in {"fp4", "nvfp4", "mxfp4"}:
        if traits.is_nvidia:
            fp4_hint = _extract_tp_from_command(
                str(model.serving.get("nvidia_nvfp4") or model.serving.get("nvidia_fp4") or "")
            )
            if fp4_hint is not None:
                hints.append((fp4_hint, "Using Blackwell FP4 launch hint published with the model."))
        elif traits.is_amd:
            fp4_hint = _extract_tp_from_command(str(model.serving.get("amd_mxfp4") or ""))
            if fp4_hint is not None:
                hints.append((fp4_hint, "Using AMD MXFP4 launch hint published with the model."))

    tp_min = model.serving.get("tp_min")
    if isinstance(tp_min, int) and tp_min > 1:
        hints.append((tp_min, f"Respecting model minimum TP requirement (tp_min={tp_min})."))

    for hint_tp, reason in hints:
        resolved = _fit_hint_to_valid_tps(hint_tp, valid_tps)
        if resolved is not None and _tp_fits_memory(model, gpu, resolved, normalized_precision):
            return resolved, reason

    return None, None


def _valid_tps(model: ModelVariant, num_gpus: int) -> list[int]:
    valid: list[int] = []
    for tp in (1, 2, 4, 8, 16, 32):
        if tp > num_gpus:
            break
        if model.kv_heads > 0 and model.kv_heads % tp != 0:
            continue
        valid.append(tp)
    return valid or [1]


def _fit_hint_to_valid_tps(hint_tp: int, valid_tps: list[int]) -> int | None:
    if hint_tp in valid_tps:
        return hint_tp
    larger = [tp for tp in valid_tps if tp >= hint_tp]
    if larger:
        return min(larger)
    return None


def _extract_tp_from_command(value: str) -> int | None:
    if not value:
        return None
    match = re.search(r"(?:^|\s)-tp\s+(\d+)(?:\s|$)", value)
    if match is None:
        return None
    return int(match.group(1))


def _tp_fits_memory(model: ModelVariant, gpu: GPUProfile, tp: int, precision: str) -> bool:
    normalized_precision = "fp4" if precision in {"nvfp4", "mxfp4"} else precision
    per_gpu_target = gpu.memory_gb * 0.90 * 0.8
    return model.weight_gb(normalized_precision) / max(tp, 1) <= per_gpu_target
