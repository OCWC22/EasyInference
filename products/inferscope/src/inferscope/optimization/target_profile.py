"""Canonical product scope for the Dynamo long-context coding lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inferscope.hardware.gpu_profiles import GPUProfile
    from inferscope.models.registry import ModelVariant


def _normalize(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _compact(value: str) -> str:
    return "".join(ch for ch in value.lower() if ch.isalnum())


@dataclass(frozen=True)
class ProductTargetProfile:
    """Immutable public support boundary for InferScope."""

    name: str
    engines: tuple[str, ...]
    models: tuple[str, ...]
    gpu_aliases: tuple[str, ...]
    workload_classes: tuple[str, ...]
    topology_modes: tuple[str, ...]
    cache_strategies: tuple[str, ...]


PRODUCT_TARGET_PROFILE = ProductTargetProfile(
    name="dynamo_long_context_coding",
    engines=("dynamo",),
    models=("Kimi-K2.5",),
    gpu_aliases=(
        "h100",
        "h100_sxm",
        "h100_sxm_80gb",
        "h100_nvl",
        "h100_nvl_94gb",
        "h100_pcie",
        "h200",
        "h200_sxm",
        "h200_sxm_141gb",
        "h200_nvl",
        "b200",
        "b300",
        "b300_ultra",
    ),
    workload_classes=("coding", "coding_long_context"),
    topology_modes=("single_endpoint", "prefill_decode_split"),
    cache_strategies=("lmcache",),
)

_TARGET_ENGINE_KEYS = {_normalize(value) for value in PRODUCT_TARGET_PROFILE.engines}
_TARGET_MODEL_KEYS = {_compact(value) for value in PRODUCT_TARGET_PROFILE.models}
_TARGET_GPU_KEYS = {_normalize(value) for value in PRODUCT_TARGET_PROFILE.gpu_aliases}
_TARGET_WORKLOAD_KEYS = {_normalize(value) for value in PRODUCT_TARGET_PROFILE.workload_classes}
_TARGET_TOPOLOGY_KEYS = {_normalize(value) for value in PRODUCT_TARGET_PROFILE.topology_modes}
_TARGET_CACHE_KEYS = {_normalize(value) for value in PRODUCT_TARGET_PROFILE.cache_strategies}


def supported_engine_names() -> list[str]:
    return list(PRODUCT_TARGET_PROFILE.engines)


def supported_model_names() -> list[str]:
    return list(PRODUCT_TARGET_PROFILE.models)


def supported_gpu_aliases() -> list[str]:
    return list(PRODUCT_TARGET_PROFILE.gpu_aliases)


def normalize_target_workload_class(value: str) -> str:
    normalized = _normalize(value)
    if normalized in {"coding_long_context", "coding"}:
        return "coding"
    return normalized


def is_target_engine(engine: str) -> bool:
    return _normalize(engine) in _TARGET_ENGINE_KEYS


def is_target_model(model: ModelVariant | str | None) -> bool:
    if model is None:
        return False
    name = model.name if hasattr(model, "name") else str(model)
    return _compact(name) in _TARGET_MODEL_KEYS


def is_target_gpu(gpu: GPUProfile | str | None) -> bool:
    if gpu is None:
        return False
    name = gpu.name if hasattr(gpu, "name") else str(gpu)
    normalized = _normalize(name)
    compact = _compact(name)
    return normalized in _TARGET_GPU_KEYS or compact in {_compact(value) for value in _TARGET_GPU_KEYS}


def is_target_workload(value: str) -> bool:
    return normalize_target_workload_class(value) in {"coding"}


def supports_topology(mode: str) -> bool:
    return _normalize(mode) in _TARGET_TOPOLOGY_KEYS


def supports_cache_strategy(strategy: str) -> bool:
    return _normalize(strategy) in _TARGET_CACHE_KEYS


def target_profile_summary() -> str:
    return (
        "InferScope production scope: Kimi-K2.5 with Dynamo + LMCache for long-context coding on H100/H200/B200/B300."
    )
