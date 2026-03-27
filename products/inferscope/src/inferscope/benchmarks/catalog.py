"""Packaged workload and experiment catalogs plus artifact comparison helpers."""

from __future__ import annotations

import json
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

from inferscope.benchmarks.experiments import BenchmarkExperimentSpec
from inferscope.benchmarks.models import BenchmarkArtifact, WorkloadPack
from inferscope.benchmarks.procedural import (
    ProceduralWorkloadOptions,
    materialize_procedural_workload,
)
from inferscope.logging import get_logger

_RESOURCE_EXTENSIONS = (".yaml", ".yml", ".json")
_LEGACY_RESOURCE_PREFIXES: dict[str, tuple[str, ...]] = {
    "workload": (
        "benchmarks/workloads/",
        "src/inferscope/benchmarks/workloads/",
    ),
    "experiment": (
        "benchmarks/experiment_specs/",
        "src/inferscope/benchmarks/experiment_specs/",
    ),
}
_MODEL_CLASS_ALIASES: dict[str, str] = {
    "dense": "dense_gqa",
    "dense_gqa": "dense_gqa",
    "reasoning": "frontier_mla_moe",
    "moe": "frontier_mla_moe",
    "frontier_mla_moe": "frontier_mla_moe",
    "coder": "compact_agentic_moe",
    "compact_agentic_moe": "compact_agentic_moe",
    "qwen": "qwen35_hybrid",
    "qwen35": "qwen35_hybrid",
    "qwen35_hybrid": "qwen35_hybrid",
    "classical_moe": "classical_moe",
}

_catalog_log = get_logger(component="benchmark_catalog")


def _find_packaged_resource(package: str, builtin_name: str) -> Path | None:
    resource_root = files(package)
    for extension in _RESOURCE_EXTENSIONS:
        candidate = resource_root.joinpath(f"{builtin_name}{extension}")
        if candidate.is_file():
            with as_file(candidate) as packaged_file:
                return packaged_file.resolve()
    return None


def _normalize_reference(reference: str | Path) -> str:
    normalized = str(reference).strip().replace("\\", "/")
    if normalized.startswith("./"):
        return normalized[2:]
    return normalized


def _normalize_identifier(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _normalize_model_class(value: str) -> str:
    normalized = _normalize_identifier(value)
    return _MODEL_CLASS_ALIASES.get(normalized, normalized)


def _legacy_builtin_name(reference: str | Path, *, kind: str) -> str | None:
    normalized = _normalize_reference(reference)
    for prefix in _LEGACY_RESOURCE_PREFIXES[kind]:
        if not normalized.startswith(prefix):
            continue
        candidate = normalized[len(prefix) :]
        resource_path = Path(candidate)
        if resource_path.suffix.lower() not in _RESOURCE_EXTENSIONS:
            return None
        if resource_path.parent != Path("."):
            return None
        return resource_path.stem
    return None


def _resolve_packaged_resource(package: str, reference: str | Path, kind: str) -> Path:
    path = Path(reference)
    if path.exists():
        return path.resolve()

    builtin_name = str(reference).strip()
    if not builtin_name:
        raise ValueError(f"{kind} reference must not be empty")

    packaged = _find_packaged_resource(package, builtin_name)
    if packaged is not None:
        return packaged

    legacy_name = _legacy_builtin_name(reference, kind=kind)
    if legacy_name is not None:
        packaged = _find_packaged_resource(package, legacy_name)
        if packaged is not None:
            return packaged

    available = ", ".join(_list_packaged_resources(package))
    raise ValueError(f"Unknown {kind} reference '{reference}'. Available built-ins: {available}")


def _list_packaged_resources(package: str) -> list[str]:
    resource_root = files(package)
    names: set[str] = set()
    for item in resource_root.iterdir():
        if item.is_file() and item.name.endswith(_RESOURCE_EXTENSIONS):
            names.add(Path(item.name).stem)
    return sorted(names)


def resolve_workload_reference(reference: str | Path) -> Path:
    """Resolve a workload reference from a file path or packaged built-in name."""
    return _resolve_packaged_resource("inferscope.benchmarks.workloads", reference, "workload")


def load_workload(reference: str | Path) -> WorkloadPack:
    """Load a workload pack from a file path or built-in name."""
    return WorkloadPack.from_file(resolve_workload_reference(reference))


def materialize_workload(
    reference: str | Path,
    *,
    options: ProceduralWorkloadOptions | None = None,
) -> WorkloadPack:
    """Resolve a workload reference and optionally expand it procedurally."""
    if options is None or not options.enabled:
        return load_workload(reference)
    if Path(reference).exists():
        raise ValueError("Procedural generation is supported only for packaged built-in workloads, not explicit files")
    seed_pack = load_workload(reference)
    return materialize_procedural_workload(seed_pack, options)


def list_builtin_workloads() -> list[str]:
    """List packaged built-in workload packs."""
    return _list_packaged_resources("inferscope.benchmarks.workloads")


def resolve_experiment_reference(reference: str | Path) -> Path:
    """Resolve an experiment reference from a file path or packaged built-in name."""
    return _resolve_packaged_resource("inferscope.benchmarks.experiment_specs", reference, "experiment")


def load_experiment(reference: str | Path) -> BenchmarkExperimentSpec:
    """Load a benchmark experiment spec from a file path or built-in name."""
    return BenchmarkExperimentSpec.from_file(resolve_experiment_reference(reference))


def list_builtin_experiments() -> list[str]:
    """List packaged built-in benchmark experiment specs."""
    return _list_packaged_resources("inferscope.benchmarks.experiment_specs")


def describe_builtin_workloads() -> list[dict[str, Any]]:
    """Return structured descriptors for all packaged workloads."""
    descriptors: list[dict[str, Any]] = []
    for name in list_builtin_workloads():
        pack = load_workload(name)
        descriptors.append(
            {
                "name": pack.name,
                "description": pack.description,
                "workload_class": pack.workload_class,
                "benchmark_role": pack.benchmark_role,
                "model": pack.model,
                "target_gpu_families": pack.target_gpu_families,
                "target_model_classes": pack.target_model_classes,
                "focus_areas": pack.focus_areas,
                "tags": pack.tags,
                "procedural": pack.name
                in {
                    "tool-agent",
                    "coding-long-context",
                    "kimi-k2-long-context-coding",
                },
            }
        )
    return descriptors


def _prefix_caching_enabled(cache_strategy: str, prefix_cache_expected: bool) -> bool:
    normalized_strategy = _normalize_identifier(cache_strategy)
    return prefix_cache_expected or "prefix" in normalized_strategy


def describe_builtin_experiments() -> list[dict[str, Any]]:
    """Return structured descriptors for all packaged experiment specs."""
    descriptors: list[dict[str, Any]] = []
    for name in list_builtin_experiments():
        try:
            spec = load_experiment(name)
            workload = load_workload(spec.workload)
        except Exception:  # noqa: BLE001
            _catalog_log.warning("skipping_invalid_experiment_descriptor", experiment=name)
            continue
        prefix_cache_expected = bool(getattr(spec.cache, "prefix_cache_expected", False))
        prefix_caching = getattr(spec.cache, "prefix_caching", None)
        descriptors.append(
            {
                "name": spec.name,
                "description": spec.description,
                "engine": spec.engine,
                "workload": spec.workload,
                "workload_class": workload.workload_class,
                "benchmark_role": spec.benchmark_role,
                "target_gpu_families": spec.target_gpu_families,
                "target_model_classes": spec.target_model_classes,
                "focus_areas": spec.focus_areas,
                "topology_mode": spec.topology.mode,
                "cache_strategy": spec.cache.strategy,
                "cache_tiers": spec.cache.tiers,
                "prefix_caching": (
                    prefix_caching
                    if isinstance(prefix_caching, bool)
                    else _prefix_caching_enabled(spec.cache.strategy, prefix_cache_expected)
                ),
                "prefix_cache_expected": prefix_cache_expected,
                "session_routing": getattr(spec.topology, "session_routing", None),
                "session_header_name": getattr(spec.topology, "session_header_name", None),
                "tags": spec.tags,
            }
        )
    return descriptors


def build_benchmark_matrix(
    *,
    gpu_family: str = "",
    model_class: str = "",
    workload_class: str = "",
    focus_area: str = "",
    engine: str = "",
) -> dict[str, Any]:
    """Return filtered benchmark workload/experiment descriptors and suggested pairings."""

    normalized_gpu = _normalize_identifier(gpu_family)
    normalized_model = _normalize_model_class(model_class)
    normalized_workload = _normalize_identifier(workload_class)
    normalized_focus = _normalize_identifier(focus_area)
    normalized_engine = _normalize_identifier(engine)

    workloads = [
        descriptor
        for descriptor in describe_builtin_workloads()
        if _matches_matrix_filters(
            descriptor,
            gpu_family=normalized_gpu,
            model_class=normalized_model,
            workload_class=normalized_workload,
            focus_area=normalized_focus,
        )
    ]
    experiments = [
        descriptor
        for descriptor in describe_builtin_experiments()
        if _matches_matrix_filters(
            descriptor,
            gpu_family=normalized_gpu,
            model_class=normalized_model,
            workload_class=normalized_workload,
            focus_area=normalized_focus,
            engine=normalized_engine,
        )
    ]
    if normalized_engine:
        workload_names_from_experiments = {
            _normalize_identifier(str(descriptor["workload"])) for descriptor in experiments
        }
        workloads = [
            descriptor
            for descriptor in workloads
            if _normalize_identifier(str(descriptor["name"])) in workload_names_from_experiments
        ]

    workload_names = {descriptor["name"] for descriptor in workloads}
    suggested_pairs = [
        {
            "experiment": descriptor["name"],
            "workload": descriptor["workload"],
            "engine": descriptor["engine"],
            "topology_mode": descriptor["topology_mode"],
            "cache_strategy": descriptor["cache_strategy"],
            "prefix_caching": descriptor.get("prefix_caching"),
            "prefix_cache_expected": descriptor.get("prefix_cache_expected"),
            "session_routing": descriptor.get("session_routing"),
            "session_header_name": descriptor.get("session_header_name"),
            "focus_areas": descriptor["focus_areas"],
        }
        for descriptor in experiments
        if descriptor["workload"] in workload_names
    ]

    return {
        "filters": {
            "gpu_family": normalized_gpu,
            "model_class": normalized_model,
            "workload_class": normalized_workload,
            "focus_area": normalized_focus,
            "engine": normalized_engine,
        },
        "workloads": workloads,
        "experiments": experiments,
        "suggested_pairs": suggested_pairs,
    }


def _matches_matrix_filters(
    descriptor: dict[str, Any],
    *,
    gpu_family: str = "",
    model_class: str = "",
    workload_class: str = "",
    focus_area: str = "",
    engine: str = "",
) -> bool:
    normalized_gpu_values = [_normalize_identifier(str(value)) for value in descriptor.get("target_gpu_families", [])]
    if gpu_family and gpu_family not in normalized_gpu_values:
        return False
    normalized_model_values = [
        _normalize_model_class(str(value)) for value in descriptor.get("target_model_classes", [])
    ]
    if model_class and model_class not in normalized_model_values:
        return False
    resolved_workload_class = _normalize_identifier(str(descriptor.get("workload_class", "")))
    resolved_workload_name = _normalize_identifier(str(descriptor.get("workload", descriptor.get("name", ""))))
    if workload_class and workload_class not in {resolved_workload_class, resolved_workload_name}:
        return False
    normalized_focus_values = [_normalize_identifier(str(value)) for value in descriptor.get("focus_areas", [])]
    if focus_area and focus_area not in normalized_focus_values:
        return False
    return not engine or engine == _normalize_identifier(str(descriptor.get("engine", "")))


def load_benchmark_artifact(path: str | Path) -> BenchmarkArtifact:
    """Load a benchmark artifact from JSON."""
    file_path = Path(path)
    data = json.loads(file_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Benchmark artifact JSON must contain an object at the top level")
    return BenchmarkArtifact.model_validate(data)


def _delta(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None:
        return None
    return new_value - base_value


def _ratio(new_value: float | None, base_value: float | None) -> float | None:
    if new_value is None or base_value is None or base_value == 0:
        return None
    return new_value / base_value


def _run_plan_field(artifact: BenchmarkArtifact, field: str, default: Any) -> Any:
    if not artifact.run_plan:
        return default
    return artifact.run_plan.get(field, default)


def _observed_runtime(artifact: BenchmarkArtifact) -> dict[str, Any]:
    observed = _run_plan_field(artifact, "observed_runtime", {})
    return observed if isinstance(observed, dict) else {}


def _runtime_metric(artifact: BenchmarkArtifact, *path: str) -> float | None:
    current: Any = _observed_runtime(artifact)
    for part in path:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    if isinstance(current, bool):
        return None
    if isinstance(current, (int, float)):
        return float(current)
    return None


def _cache_effectiveness_metric(artifact: BenchmarkArtifact, metric: str) -> float | None:
    nested_metric = _runtime_metric(artifact, "cache_effectiveness", metric)
    if nested_metric is not None:
        return nested_metric
    return _runtime_metric(artifact, metric)


def _topology_mode(artifact: BenchmarkArtifact) -> str:
    topology = _run_plan_field(artifact, "topology", {})
    if isinstance(topology, dict):
        return str(topology.get("mode", "single_endpoint"))
    return "single_endpoint"


def _cache_strategy(artifact: BenchmarkArtifact) -> str:
    cache = _run_plan_field(artifact, "cache", {})
    if isinstance(cache, dict):
        return str(cache.get("strategy", "unknown"))
    return "unknown"


def _metrics_roles(artifact: BenchmarkArtifact) -> list[str]:
    if artifact.run_plan and isinstance(artifact.run_plan.get("metrics_targets"), list):
        role_list = [
            str(target.get("role", "primary"))
            for target in artifact.run_plan["metrics_targets"]
            if isinstance(target, dict)
        ]
        return sorted(role_list) if role_list else ["primary"]
    if artifact.metrics_before_targets or artifact.metrics_after_targets:
        combined_snapshots = artifact.metrics_before_targets + artifact.metrics_after_targets
        role_set = {snapshot.target_role for snapshot in combined_snapshots}
        return sorted(role_set) if role_set else ["primary"]
    return ["primary"]


def compare_benchmark_artifacts(
    baseline: BenchmarkArtifact,
    candidate: BenchmarkArtifact,
) -> dict[str, Any]:
    """Compare two benchmark artifacts."""
    baseline_summary = baseline.summary
    candidate_summary = candidate.summary

    compatibility_warnings: list[str] = []
    differing_fields: list[str] = []

    comparable_fields = {
        "pack_name": (baseline.pack_name, candidate.pack_name),
        "workload_class": (baseline.workload_class, candidate.workload_class),
        "model": (baseline.model, candidate.model),
        "concurrency": (baseline.concurrency, candidate.concurrency),
        "topology_mode": (_topology_mode(baseline), _topology_mode(candidate)),
        "cache_strategy": (_cache_strategy(baseline), _cache_strategy(candidate)),
        "metrics_roles": (_metrics_roles(baseline), _metrics_roles(candidate)),
    }

    for field, (baseline_value, candidate_value) in comparable_fields.items():
        if baseline_value != candidate_value:
            differing_fields.append(field)
            compatibility_warnings.append(
                f"Different {field}: baseline={baseline_value!r} candidate={candidate_value!r}"
            )

    comparison: dict[str, Any] = {
        "baseline": {
            "path": baseline.default_filename,
            "pack_name": baseline.pack_name,
            "endpoint": baseline.endpoint,
            "model": baseline.model,
            "summary": baseline_summary.model_dump(mode="json"),
        },
        "candidate": {
            "path": candidate.default_filename,
            "pack_name": candidate.pack_name,
            "endpoint": candidate.endpoint,
            "model": candidate.model,
            "summary": candidate_summary.model_dump(mode="json"),
        },
        "compatibility": {
            "comparable": not differing_fields,
            "warnings": compatibility_warnings,
            "differing_fields": differing_fields,
        },
        "deltas": {
            "latency_p95_ms": _delta(candidate_summary.latency_p95_ms, baseline_summary.latency_p95_ms),
            "ttft_p95_ms": _delta(candidate_summary.ttft_p95_ms, baseline_summary.ttft_p95_ms),
            "latency_avg_ms": _delta(candidate_summary.latency_avg_ms, baseline_summary.latency_avg_ms),
            "wall_time_ms": _delta(candidate_summary.wall_time_ms, baseline_summary.wall_time_ms),
            "succeeded": candidate_summary.succeeded - baseline_summary.succeeded,
            "failed": candidate_summary.failed - baseline_summary.failed,
            "total_tokens": candidate_summary.total_tokens - baseline_summary.total_tokens,
            "request_throughput_rps": _delta(
                _runtime_metric(candidate, "request_throughput_rps"),
                _runtime_metric(baseline, "request_throughput_rps"),
            ),
            "output_throughput_tps": _delta(
                _runtime_metric(candidate, "output_throughput_tps"),
                _runtime_metric(baseline, "output_throughput_tps"),
            ),
            "goodput_rps": _delta(
                _runtime_metric(candidate, "goodput_rps"),
                _runtime_metric(baseline, "goodput_rps"),
            ),
            "tpot_p95_ms": _delta(
                _runtime_metric(candidate, "tpot_ms", "p95"),
                _runtime_metric(baseline, "tpot_ms", "p95"),
            ),
            "itl_p95_ms": _delta(
                _runtime_metric(candidate, "itl_ms", "p95"),
                _runtime_metric(baseline, "itl_ms", "p95"),
            ),
            "tool_parse_success_rate": _delta(
                _runtime_metric(candidate, "tool_parse_success_rate"),
                _runtime_metric(baseline, "tool_parse_success_rate"),
            ),
            "prefix_cache_hit_rate": _delta(
                _cache_effectiveness_metric(candidate, "prefix_cache_hit_rate"),
                _cache_effectiveness_metric(baseline, "prefix_cache_hit_rate"),
            ),
            "prefix_cache_hits": _delta(
                _cache_effectiveness_metric(candidate, "prefix_cache_hits"),
                _cache_effectiveness_metric(baseline, "prefix_cache_hits"),
            ),
        },
        "ratios": {
            "latency_p95": _ratio(candidate_summary.latency_p95_ms, baseline_summary.latency_p95_ms),
            "ttft_p95": _ratio(candidate_summary.ttft_p95_ms, baseline_summary.ttft_p95_ms),
            "wall_time": _ratio(candidate_summary.wall_time_ms, baseline_summary.wall_time_ms),
            "request_throughput": _ratio(
                _runtime_metric(candidate, "request_throughput_rps"),
                _runtime_metric(baseline, "request_throughput_rps"),
            ),
            "output_throughput": _ratio(
                _runtime_metric(candidate, "output_throughput_tps"),
                _runtime_metric(baseline, "output_throughput_tps"),
            ),
            "goodput": _ratio(
                _runtime_metric(candidate, "goodput_rps"),
                _runtime_metric(baseline, "goodput_rps"),
            ),
        },
    }

    latency_delta = comparison["deltas"]["latency_p95_ms"]
    ttft_delta = comparison["deltas"]["ttft_p95_ms"]
    summary_parts = []
    if latency_delta is not None:
        direction = "faster" if latency_delta < 0 else "slower" if latency_delta > 0 else "unchanged"
        summary_parts.append(f"p95 latency {direction} by {abs(latency_delta):.1f} ms")
    if ttft_delta is not None:
        direction = "lower" if ttft_delta < 0 else "higher" if ttft_delta > 0 else "unchanged"
        summary_parts.append(f"p95 TTFT {direction} by {abs(ttft_delta):.1f} ms")
    if compatibility_warnings:
        summary_parts.append(f"compatibility warnings: {len(compatibility_warnings)}")
    if not summary_parts:
        summary_parts.append("comparison computed")
    comparison["summary"] = " | ".join(summary_parts)
    return comparison
