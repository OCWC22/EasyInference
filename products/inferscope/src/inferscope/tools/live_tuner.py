"""Live auto-tuner — detects failure modes and recommends config adjustments.

Scrapes a live endpoint's Prometheus metrics, runs workload-aware failure
mode checks, and returns specific parameter mutations to fix detected issues.
This is the detection + fix engine for InferScope's auto-tuning capability.

IMPORTANT: This tool never auto-applies changes. It returns recommendations
that a human or orchestrator deploys.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.optimization.checks import (
    AuditFinding,
    DeploymentContext,
    run_all_checks,
)
from inferscope.telemetry.normalizer import NormalizedMetrics, normalize
from inferscope.telemetry.prometheus import scrape_metrics


@dataclass
class TuningAdjustment:
    """A single recommended parameter change."""

    parameter: str  # e.g., "scheduler.chunked_prefill"
    current_value: Any
    recommended_value: Any
    reason: str
    confidence: float  # 0-1
    trigger: str  # check_id that triggered this

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameter": self.parameter,
            "current_value": self.current_value,
            "recommended_value": self.recommended_value,
            "reason": self.reason,
            "confidence": round(self.confidence, 2),
            "trigger": self.trigger,
        }


def _derive_adjustments(
    findings: list[AuditFinding],
    metrics: NormalizedMetrics,
    current_scheduler: dict[str, Any],
    current_cache: dict[str, Any],
) -> list[TuningAdjustment]:
    """Map detected failure modes to specific config adjustments."""
    adjustments: list[TuningAdjustment] = []
    seen_params: set[str] = set()

    for finding in findings:
        new_adjustments = _adjustments_for_finding(
            finding, metrics, current_scheduler, current_cache
        )
        for adj in new_adjustments:
            if adj.parameter not in seen_params:
                adjustments.append(adj)
                seen_params.add(adj.parameter)

    return adjustments


def _adjustments_for_finding(
    finding: AuditFinding,
    metrics: NormalizedMetrics,
    scheduler: dict[str, Any],
    cache: dict[str, Any],
) -> list[TuningAdjustment]:
    """Generate adjustments for a specific finding."""
    adjs: list[TuningAdjustment] = []
    cid = finding.check_id

    if cid == "DECODE_STARVATION":
        current_priority = scheduler.get("decode_priority", 0.5)
        adjs.append(TuningAdjustment(
            parameter="scheduler.decode_priority",
            current_value=current_priority,
            recommended_value=min(0.9, current_priority + 0.2),
            reason="Decode tokens starved by prefill batches — increase decode scheduling priority",
            confidence=0.8,
            trigger=cid,
        ))
        current_ratio = scheduler.get("max_prefill_chunk_ratio", 0.5)
        adjs.append(TuningAdjustment(
            parameter="scheduler.max_prefill_chunk_ratio",
            current_value=current_ratio,
            recommended_value=max(0.2, current_ratio - 0.15),
            reason="Limit prefill batch fraction to free scheduler budget for decode",
            confidence=0.75,
            trigger=cid,
        ))

    elif cid == "PREFILL_STARVATION":
        current_priority = scheduler.get("decode_priority", 0.5)
        adjs.append(TuningAdjustment(
            parameter="scheduler.decode_priority",
            current_value=current_priority,
            recommended_value=max(0.2, current_priority - 0.2),
            reason="Prefill queued behind heavy decode — reduce decode priority to admit new requests",
            confidence=0.8,
            trigger=cid,
        ))
        current_budget = scheduler.get("prefill_lane_budget", 0)
        target_budget = scheduler.get("batched_token_budget", 8192)
        adjs.append(TuningAdjustment(
            parameter="scheduler.prefill_lane_budget",
            current_value=current_budget,
            recommended_value=target_budget // 2,
            reason="Reserve dedicated prefill budget to prevent decode from starving new requests",
            confidence=0.7,
            trigger=cid,
        ))

    elif cid == "PCIE_OFFLOAD_THRASH":
        adjs.append(TuningAdjustment(
            parameter="cache.offload_policy",
            current_value=cache.get("offload_policy", "cold_only"),
            recommended_value="disabled",
            reason="PCIe transfer during active decode causes latency explosion — disable offloading",
            confidence=0.85,
            trigger=cid,
        ))

    elif cid == "GPU_UNDERUTILIZATION":
        current_seqs = scheduler.get("max_num_seqs", 256)
        adjs.append(TuningAdjustment(
            parameter="scheduler.max_num_seqs",
            current_value=current_seqs,
            recommended_value=min(512, current_seqs * 2),
            reason="GPU has KV headroom but scheduler is limiting admissions",
            confidence=0.8,
            trigger=cid,
        ))
        current_budget = scheduler.get("batched_token_budget", 8192)
        adjs.append(TuningAdjustment(
            parameter="scheduler.batched_token_budget",
            current_value=current_budget,
            recommended_value=min(32768, current_budget * 2),
            reason="Raise batch budget to admit more tokens per step while GPU has capacity",
            confidence=0.75,
            trigger=cid,
        ))

    elif cid == "OOM_DESPITE_FREE":
        adjs.append(TuningAdjustment(
            parameter="cache.fragmentation_check",
            current_value=cache.get("fragmentation_check", False),
            recommended_value=True,
            reason="Preemptions despite free KV — enable fragmentation monitoring",
            confidence=0.85,
            trigger=cid,
        ))
        adjs.append(TuningAdjustment(
            parameter="cache.kv_compaction_trigger",
            current_value=cache.get("kv_compaction_trigger", 0.4),
            recommended_value=0.3,
            reason="Lower compaction trigger to reclaim fragmented blocks earlier",
            confidence=0.75,
            trigger=cid,
        ))

    elif cid == "KV_FRAGMENTATION_HIGH":
        adjs.append(TuningAdjustment(
            parameter="cache.kv_compaction_trigger",
            current_value=cache.get("kv_compaction_trigger", 0.4),
            recommended_value=0.5,
            reason="High KV usage with low active sequences — raise compaction trigger",
            confidence=0.7,
            trigger=cid,
        ))

    elif cid == "HIGH_TTFT":
        # TTFT spike: consider disabling chunked prefill or increasing prefill budget
        current_chunked = scheduler.get("chunked_prefill", True)
        if current_chunked:
            adjs.append(TuningAdjustment(
                parameter="scheduler.chunked_prefill",
                current_value=True,
                recommended_value=False,
                reason="High TTFT may be caused by chunked prefill splitting — try contiguous prefill",
                confidence=0.65,
                trigger=cid,
            ))

    elif cid == "KV_CACHE_CRITICAL":
        current_util = cache.get("gpu_memory_utilization", 0.92)
        adjs.append(TuningAdjustment(
            parameter="cache.gpu_memory_utilization",
            current_value=current_util,
            recommended_value=max(0.85, current_util - 0.03),
            reason="KV cache saturated — lower memory utilization to create headroom",
            confidence=0.8,
            trigger=cid,
        ))

    return adjs


def _apply_adjustments(
    adjustments: list[TuningAdjustment],
    scheduler: dict[str, Any],
    cache: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Apply adjustments to copies of scheduler/cache dicts (for preview)."""
    updated_scheduler = dict(scheduler)
    updated_cache = dict(cache)

    for adj in adjustments:
        prefix, key = adj.parameter.split(".", 1)
        if prefix == "scheduler":
            updated_scheduler[key] = adj.recommended_value
        elif prefix == "cache":
            updated_cache[key] = adj.recommended_value

    return updated_scheduler, updated_cache


async def auto_tune_deployment(
    endpoint: str,
    current_engine: str = "",
    current_workload: str = "",
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict[str, Any]:
    """Analyze a live endpoint and recommend config adjustments.

    Detects: TTFT spikes, throughput collapse, GPU underutilization,
    PCIe bottlenecks, KV fragmentation, decode/prefill starvation.
    Returns specific parameter changes to fix detected issues.

    This tool NEVER auto-applies changes — it returns recommendations only.

    Args:
        endpoint: Running inference endpoint URL (e.g., http://localhost:8000)
        current_engine: Engine type if known (vllm, sglang, atom)
        current_workload: Workload mode if known (coding, chat, agent, long_context_rag)
        current_scheduler: Current scheduler config dict (for delta computation)
        current_cache: Current cache config dict (for delta computation)
        allow_private: Allow localhost/private IPs
        metrics_auth: Optional auth config for metrics endpoint
    """
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)

    if scrape.error:
        return {
            "error": scrape.error,
            "endpoint": endpoint,
            "summary": f"Cannot reach {endpoint}/metrics — {scrape.error}",
            "confidence": 0.0,
        }

    metrics = normalize(scrape)

    # Build deployment context from what we know
    ctx = DeploymentContext(
        engine=current_engine or metrics.engine,
    )

    # Run all checks (including the 6 new workload-aware checks)
    findings = run_all_checks(metrics, ctx)

    # Default scheduler/cache if not provided
    sched = current_scheduler or {
        "batched_token_budget": 8192,
        "max_num_seqs": 256,
        "decode_priority": 0.5,
        "chunked_prefill": True,
        "prefill_decode_isolation": "colocated",
        "prefill_lane_budget": 0,
        "decode_lane_budget": 0,
        "max_prefill_chunk_ratio": 0.5,
    }
    cache = current_cache or {
        "gpu_memory_utilization": 0.92,
        "offload_policy": "cold_only",
        "kv_compaction_trigger": 0.4,
        "fragmentation_check": False,
        "pcie_utilization_cap": 0.7,
    }

    # Derive adjustments from findings
    adjustments = _derive_adjustments(findings, metrics, sched, cache)

    # Preview updated config
    updated_scheduler, updated_cache = _apply_adjustments(adjustments, sched, cache)

    # Build reasoning trace
    reasoning = []
    if not findings:
        reasoning.append("No failure modes detected — current config appears healthy.")
    for f in findings:
        reasoning.append(f"Detected {f.check_id} ({f.severity}): {f.title}")
    for adj in adjustments:
        reasoning.append(
            f"Adjustment: {adj.parameter} {adj.current_value} → {adj.recommended_value} "
            f"(trigger: {adj.trigger}, confidence: {adj.confidence:.0%})"
        )

    return {
        "detections": [f.to_dict() for f in findings],
        "adjustments": [a.to_dict() for a in adjustments],
        "updated_scheduler": updated_scheduler,
        "updated_cache": updated_cache,
        "reasoning": reasoning,
        "metrics_snapshot": metrics.to_dict(),
        "summary": (
            f"{len(findings)} issue(s) detected, {len(adjustments)} adjustment(s) recommended"
            if findings
            else "No issues detected — deployment appears healthy"
        ),
        "confidence": 0.85 if adjustments else 0.9,
        "evidence": "live_metrics_analysis",
    }
