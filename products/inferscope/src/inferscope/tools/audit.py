"""Live deployment audit — runs all checks against a running endpoint.

This is the flagship tool: point it at a vLLM/SGLang/ATOM endpoint and
get ranked, ISA-grounded findings with exact fix commands.
"""

from __future__ import annotations

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.logging import get_logger, sanitize_log_text
from inferscope.optimization.checks import DeploymentContext, run_all_checks
from inferscope.optimization.workload_classifier import classify_workload
from inferscope.telemetry.normalizer import normalize
from inferscope.telemetry.prometheus import scrape_metrics

log = get_logger(component="audit")


async def audit_deployment(
    endpoint: str,
    gpu_arch: str = "",
    gpu_name: str = "",
    model_name: str = "",
    model_type: str = "",
    attention_type: str = "",
    experts_total: int = 0,
    tp: int = 1,
    ep: int = 0,
    quantization: str = "",
    kv_cache_dtype: str = "",
    gpu_memory_utilization: float = 0.0,
    block_size: int = 0,
    has_rdma: bool = False,
    split_prefill_decode: bool = False,
    allow_private: bool = True,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Run all audit checks against a live inference endpoint.

    Auto-detects engine type from Prometheus metrics. GPU architecture
    and model details can be provided for richer checks, or left empty
    for basic metric-only checks.

    Returns ranked findings with ISA-grounded recommendations.
    """
    audit_log = log.bind(
        endpoint=endpoint,
        gpu_arch=gpu_arch,
        gpu_name=gpu_name,
        model_name=model_name,
        allow_private=allow_private,
    )
    audit_log.info(
        "audit_started",
        split_prefill_decode=split_prefill_decode,
        has_rdma=has_rdma,
        tp=tp,
        ep=ep,
    )

    # 1. Scrape metrics
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)
    if scrape.error:
        audit_log.error(
            "audit_failed",
            error_type="scrape_error",
            error_summary=sanitize_log_text(scrape.error),
        )
        return {
            "error": scrape.error,
            "endpoint": endpoint,
            "summary": f"❌ Audit failed: {scrape.error}",
            "confidence": 0.0,
            "evidence": "scrape_failure",
        }

    # 2. Normalize
    metrics = normalize(scrape)

    # 3. Classify workload
    workload = classify_workload(metrics)

    # 4. Build deployment context from user-provided + detected info
    is_amd = gpu_arch.startswith("gfx") if gpu_arch else False
    ctx = DeploymentContext(
        engine=metrics.engine,
        gpu_arch=gpu_arch,
        gpu_name=gpu_name,
        gpu_vendor="amd" if is_amd else ("nvidia" if gpu_arch.startswith("sm") else ""),
        model_name=model_name,
        model_type=model_type,
        attention_type=attention_type,
        experts_total=experts_total,
        tp=tp,
        ep=ep,
        fp8_support=gpu_arch in ("sm_90a", "sm_90", "sm_100", "sm_103", "gfx942", "gfx950"),
        fp8_format=(
            "OCP" if gpu_arch in ("sm_90a", "sm_90", "sm_100", "sm_103", "gfx950")
            else ("FNUZ" if gpu_arch == "gfx942" else "")
        ),
        gpu_memory_utilization=gpu_memory_utilization,
        kv_cache_dtype=kv_cache_dtype,
        quantization=quantization,
        block_size=block_size,
        has_rdma=has_rdma,
        split_prefill_decode=split_prefill_decode,
        # Try to detect env vars from context
        env_vars={},
    )

    # 5. Run all checks
    findings = run_all_checks(metrics, ctx)

    # 6. Build response
    critical_count = sum(1 for f in findings if f.severity == "critical")
    warning_count = sum(1 for f in findings if f.severity == "warning")
    info_count = sum(1 for f in findings if f.severity == "info")

    audit_log.info(
        "audit_completed",
        engine=metrics.engine,
        total_findings=len(findings),
        critical=critical_count,
        warnings=warning_count,
        info=info_count,
        workload_mode=workload.mode.value,
        workload_confidence=round(workload.confidence, 3),
    )

    return {
        "audit": {
            "findings": [f.to_dict() for f in findings],
            "total": len(findings),
            "critical": critical_count,
            "warnings": warning_count,
            "info": info_count,
        },
        "workload": workload.to_dict(),
        "metrics": metrics.to_dict(),
        "engine": metrics.engine,
        "endpoint": endpoint,
        "summary": (
            f"{metrics.engine.upper()} audit: {len(findings)} finding(s) "
            f"({critical_count} critical, {warning_count} warnings, {info_count} info) | "
            f"Workload: {workload.mode.value} ({workload.confidence:.0%} confidence)"
        ),
        "confidence": 0.85 if gpu_arch else 0.65,  # Higher if hardware context provided
        "evidence": "live_audit_checks",
    }
