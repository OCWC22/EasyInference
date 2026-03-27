"""Production benchmark and MCP contract for the narrowed InferScope surface."""

from __future__ import annotations

from typing import Any

from inferscope.benchmarks.models import BenchmarkArtifact
from inferscope.hardware.gpu_profiles import GPUProfile, get_gpu_profile
from inferscope.models.registry import ModelVariant, get_model_variant

SUPPORTED_MODEL = "Kimi-K2.5"
SUPPORTED_MODELS = (SUPPORTED_MODEL,)
SUPPORTED_ENGINE = "dynamo"
SUPPORTED_BENCHMARK_ENGINES = ("vllm", "dynamo")
SUPPORTED_WORKLOAD_MODE = "coding"
SUPPORTED_WORKLOAD_PACK = "kimi-k2-long-context-coding"
SUPPORTED_EXPERIMENTS = (
    "dynamo-aggregated-lmcache-kimi-k2",
    "vllm-disagg-prefill-lmcache",
    "dynamo-disagg-lmcache-kimi-k2",
)
SUPPORTED_GPU_CANONICAL = ("h100", "h200", "b200", "b300")
SUPPORTED_GPU_NAMES = {
    "H100 SXM",
    "H100 NVL",
    "H100 PCIe",
    "H200 SXM",
    "H200 NVL",
    "B200",
    "B300",
}

FRONTEND_METRICS = [
    "dynamo_frontend_inflight_requests",
    "dynamo_frontend_queued_requests",
    "dynamo_frontend_time_to_first_token_seconds",
    "dynamo_frontend_inter_token_latency_seconds",
    "dynamo_frontend_request_duration_seconds",
    "dynamo_frontend_disconnected_clients",
    "dynamo_frontend_model_migration_total",
]
BACKEND_METRICS = [
    "dynamo_component_inflight_requests",
    "dynamo_component_request_duration_seconds",
    "dynamo_component_requests_total",
]
KV_METRICS = [
    "dynamo_component_kvstats_gpu_cache_usage_percent",
    "dynamo_component_kvstats_gpu_prefix_cache_hit_rate",
    "dynamo_component_kvstats_active_blocks",
    "dynamo_component_kvstats_total_blocks",
]
TRACE_ENV_VARS = [
    "DYN_LOGGING_JSONL",
    "OTEL_EXPORT_ENABLED",
    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
    "OTEL_SERVICE_NAME",
]
REQUEST_HEADERS = ["x-request-id", "x-session-id"]


def resolve_supported_model(model_name: str) -> ModelVariant | None:
    """Resolve the only supported production model."""
    if not model_name.strip():
        return None
    model = get_model_variant(model_name)
    if model is None or model.name != SUPPORTED_MODEL:
        return None
    return model


def resolve_supported_gpu(gpu_name: str) -> GPUProfile | None:
    """Resolve a supported Hopper or Blackwell production GPU."""
    if not gpu_name.strip():
        return None
    gpu = get_gpu_profile(gpu_name)
    if gpu is None or gpu.name not in SUPPORTED_GPU_NAMES:
        return None
    return gpu


def _minimum_tp_for_lane(model: ModelVariant) -> int:
    hinted = model.serving.get("tp_min")
    if isinstance(hinted, int) and hinted > 0:
        return hinted
    return 1


def _minimum_tp_for_gpu(model: ModelVariant, gpu: GPUProfile | None) -> int:
    if gpu is None:
        for key in ("tp_fp8_h200", "tp_fp8_b200", "tp_fp8_b300", "tp_fp8_h100"):
            hinted = model.serving.get(key)
            if isinstance(hinted, int) and hinted > 0:
                return hinted
        return _minimum_tp_for_lane(model)

    name = gpu.name.lower()
    candidate_keys: list[str] = []
    if "h100" in name:
        candidate_keys.append("tp_fp8_h100")
    elif "h200" in name:
        candidate_keys.append("tp_fp8_h200")
    elif gpu.name == "B200":
        if gpu.fp4_support:
            candidate_keys.append("tp_fp4_b200")
        candidate_keys.append("tp_fp8_b200")
    elif gpu.name == "B300":
        if gpu.fp4_support:
            candidate_keys.append("tp_fp4_b300")
        candidate_keys.append("tp_fp8_b300")

    for key in candidate_keys:
        hinted = model.serving.get(key)
        if isinstance(hinted, int) and hinted > 0:
            return hinted
    return _minimum_tp_for_lane(model)


def required_gpus_for_topology(
    *,
    model_name: str,
    gpu_name: str = "",
    topology_mode: str,
) -> int:
    """Return the minimum practical GPU count for the target topology."""
    model = resolve_supported_model(model_name) or resolve_supported_model(SUPPORTED_MODEL)
    gpu = resolve_supported_gpu(gpu_name) if gpu_name else None
    tp_min = _minimum_tp_for_gpu(model, gpu) if model is not None else 1
    normalized_topology = topology_mode.strip().lower() or "single_endpoint"
    if normalized_topology == "prefill_decode_split":
        return tp_min * 2
    return tp_min


def validate_production_target(
    *,
    model_name: str = "",
    gpu_name: str = "",
    workload: str = "",
    engine: str = "",
    num_gpus: int = 0,
    topology_mode: str = "",
) -> list[str]:
    """Return user-facing validation errors for unsupported MCP benchmark targets."""
    errors: list[str] = []

    if model_name and resolve_supported_model(model_name) is None:
        errors.append(f"InferScope MCP is currently scoped to model '{SUPPORTED_MODEL}' only; got '{model_name}'.")

    if gpu_name and resolve_supported_gpu(gpu_name) is None:
        errors.append(
            "InferScope MCP currently supports Hopper/Blackwell production GPUs only: H100, H200, B200, B300."
        )

    if workload:
        normalized = workload.strip().lower().replace("-", "_")
        if normalized not in {"coding", "coding_agent", "long_context_coding"}:
            errors.append("InferScope MCP is currently scoped to long-context coding workloads only.")

    if engine:
        normalized_engine = engine.strip().lower()
        if normalized_engine not in {"", "auto", *SUPPORTED_BENCHMARK_ENGINES}:
            errors.append("InferScope MCP currently supports vLLM and NVIDIA Dynamo only.")

    if num_gpus:
        required_gpus = required_gpus_for_topology(
            model_name=model_name or SUPPORTED_MODEL,
            gpu_name=gpu_name,
            topology_mode=topology_mode,
        )
        if required_gpus and num_gpus < required_gpus:
            topology_label = topology_mode or "single_endpoint"
            errors.append(
                f"InferScope MCP requires at least {required_gpus} GPU(s) for "
                f"{topology_label} serving on this target; got {num_gpus}."
            )

    return errors


def build_production_contract() -> dict[str, Any]:
    """Return the supported deployment, benchmarking, and observability contract."""
    return {
        "model": SUPPORTED_MODEL,
        "models": list(SUPPORTED_MODELS),
        "engine": SUPPORTED_ENGINE,
        "benchmark_engines": list(SUPPORTED_BENCHMARK_ENGINES),
        "workload_mode": SUPPORTED_WORKLOAD_MODE,
        "workload_pack": SUPPORTED_WORKLOAD_PACK,
        "experiments": list(SUPPORTED_EXPERIMENTS),
        "supported_gpu_aliases": list(SUPPORTED_GPU_CANONICAL),
        "observability": {
            "frontend_metrics": list(FRONTEND_METRICS),
            "backend_metrics": list(BACKEND_METRICS),
            "kv_metrics": list(KV_METRICS),
            "trace_env_vars": list(TRACE_ENV_VARS),
            "request_headers": list(REQUEST_HEADERS),
            "backend_system_port_env": "DYN_SYSTEM_PORT",
            "summary": (
                "Scrape frontend and worker metrics separately. Frontend exposes queue, "
                "TTFT, ITL, disconnects, and migration counters; worker system ports "
                "expose request duration and KV stats."
            ),
        },
        "reliability": {
            "minimum_success_rate": 0.99,
            "require_metrics_capture_complete": True,
            "max_failed_sessions": 0,
            "warning_queue_depth": 10,
            "warning_migrations": 1,
            "warning_kv_usage": 0.9,
        },
        "topologies": [
            {
                "name": "aggregated",
                "experiment": SUPPORTED_EXPERIMENTS[0],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="single_endpoint",
                ),
                "summary": ("Single Dynamo frontend plus one aggregated Kimi worker group with LMCache enabled."),
            },
            {
                "name": "disaggregated_comparison",
                "experiment": SUPPORTED_EXPERIMENTS[1],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="prefill_decode_split",
                ),
                "summary": (
                    "vLLM disaggregated LMCache comparison lane for the same Kimi long-context coding workload."
                ),
            },
            {
                "name": "disaggregated_production",
                "experiment": SUPPORTED_EXPERIMENTS[2],
                "min_gpus": required_gpus_for_topology(
                    model_name=SUPPORTED_MODEL,
                    gpu_name="H200 SXM",
                    topology_mode="prefill_decode_split",
                ),
                "summary": ("Dynamo frontend plus separate prefill/decode workers with LMCache and KV-aware routing."),
            },
        ],
    }


def build_benchmark_readiness_summary(artifact: BenchmarkArtifact) -> dict[str, Any]:
    """Summarize benchmark reliability and observability in production terms."""
    total_requests = artifact.summary.total_requests or 0
    succeeded = artifact.summary.succeeded
    success_rate = (succeeded / total_requests) if total_requests else 0.0
    observed_runtime = artifact.run_plan.get("observed_runtime", {}) if artifact.run_plan else {}
    reliability = observed_runtime.get("reliability", {}) if isinstance(observed_runtime, dict) else {}
    observability = observed_runtime.get("observability", {}) if isinstance(observed_runtime, dict) else {}

    issues: list[str] = []
    if success_rate < 0.99:
        issues.append(f"Request success rate is {success_rate:.1%}, below the 99% production target.")
    if not artifact.summary.metrics_capture_complete:
        issues.append("Metrics capture was incomplete across declared targets.")
    if observability.get("missing_metric_prefixes"):
        issues.append("Required observability prefixes were missing from at least one metrics target.")
    if observability.get("targets_with_errors"):
        issues.append("At least one metrics target failed to scrape cleanly.")

    failed_sessions = int(reliability.get("failed_sessions", 0) or 0)
    if failed_sessions > 0:
        issues.append(f"{failed_sessions} session(s) failed, which breaks coding-session reliability guarantees.")

    observability_gaps = len(observability.get("missing_metric_prefixes", []))
    return {
        "ready": not issues,
        "success_rate": round(success_rate, 4),
        "metrics_capture_complete": artifact.summary.metrics_capture_complete,
        "metrics_targets_total": artifact.summary.metrics_targets_total,
        "metrics_targets_with_errors": artifact.summary.metrics_targets_with_errors,
        "failed_sessions": failed_sessions,
        "failure_types": reliability.get("failure_types", {}),
        "missing_metric_prefixes": observability.get("missing_metric_prefixes", []),
        "issues": issues,
        "summary": (
            f"success={success_rate:.1%}, "
            f"metrics_complete={artifact.summary.metrics_capture_complete}, "
            f"failed_sessions={failed_sessions}, "
            f"observability_gaps={observability_gaps}"
        ),
    }
