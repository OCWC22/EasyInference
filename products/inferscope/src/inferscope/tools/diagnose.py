"""Live deployment diagnostics — scrape real Prometheus metrics and analyze.

These tools require a running vLLM/SGLang/ATOM endpoint.
"""

from __future__ import annotations

from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.telemetry.normalizer import NormalizedMetrics, normalize
from inferscope.telemetry.prometheus import scrape_metrics


async def check_deployment(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Scrape a live inference endpoint and return normalized health snapshot.

    Works with vLLM, SGLang, and ATOM — auto-detects which engine is running.
    Returns structured metrics with health indicators.

    Args:
        allow_private: Allow localhost/private IPs. True for CLI, False for MCP.
    """
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)

    if scrape.error:
        return {
            "error": scrape.error,
            "endpoint": endpoint,
            "summary": f"❌ Cannot reach {endpoint}/metrics — {scrape.error}",
            "confidence": 0.0,
            "evidence": "scrape_failure",
        }

    metrics = normalize(scrape)
    health = _assess_health(metrics)

    return {
        "metrics": metrics.to_dict(),
        "health": health,
        "summary": _build_summary(metrics, health),
        "confidence": 0.9 if metrics.engine != "unknown" else 0.5,
        "evidence": "live_prometheus_scrape",
    }


async def check_memory_pressure(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Analyze KV cache utilization and preemption rates from live metrics."""
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)

    if scrape.error:
        return {"error": scrape.error, "confidence": 0.0}

    metrics = normalize(scrape)

    kv_usage = metrics.kv_cache_usage
    preemptions = metrics.preemptions_total
    prefix_hits = metrics.prefix_cache_hit_rate

    pressure = "low"
    findings = []

    if kv_usage > 0.95:
        pressure = "critical"
        findings.append(
            f"KV cache at {kv_usage:.0%} — preemptions imminent. "
            "Reduce gpu_memory_utilization, lower max_model_len, or add replicas."
        )
    elif kv_usage > 0.85:
        pressure = "high"
        findings.append(
            f"KV cache at {kv_usage:.0%} — approaching saturation. "
            "Consider FP8 KV cache (2x savings) or CPU offloading."
        )
    elif kv_usage > 0.70:
        pressure = "moderate"
        findings.append(f"KV cache at {kv_usage:.0%} — healthy utilization.")
    else:
        findings.append(
            f"KV cache at {kv_usage:.0%} — underutilized. "
            "Could increase gpu_memory_utilization or serve more concurrent requests."
        )

    if preemptions > 0:
        findings.append(
            f"⚠ {preemptions:.0f} preemptions recorded — requests being evicted from KV cache under pressure."
        )

    if prefix_hits > 0:
        findings.append(f"Prefix cache hit rate: {prefix_hits:.0%}")

    return {
        "memory_pressure": {
            "level": pressure,
            "kv_cache_usage": round(kv_usage, 4),
            "prefix_cache_hit_rate": round(prefix_hits, 4),
            "preemptions_total": preemptions,
            "cpu_cache_usage": round(metrics.cpu_cache_usage, 4),
        },
        "findings": findings,
        "engine": metrics.engine,
        "summary": f"Memory pressure: {pressure} (KV {kv_usage:.0%}, {len(findings)} findings)",
        "confidence": 0.85,
        "evidence": "live_kv_cache_metrics",
    }


async def get_cache_effectiveness(
    endpoint: str,
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
) -> dict:
    """Measure prefix cache hit rate and cache-aware routing effectiveness."""
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)

    if scrape.error:
        return {"error": scrape.error, "confidence": 0.0}

    metrics = normalize(scrape)

    hit_rate = metrics.prefix_cache_hit_rate
    kv_usage = metrics.kv_cache_usage

    effectiveness = "unknown"
    recommendations = []

    if hit_rate > 0.8:
        effectiveness = "excellent"
        recommendations.append("High cache hit rate — prefix caching is working well.")
    elif hit_rate > 0.5:
        effectiveness = "good"
        recommendations.append(
            "Moderate cache hits — consider canonicalizing prompts "
            "(remove timestamps, request IDs, tool noise from prefix)."
        )
    elif hit_rate > 0.2:
        effectiveness = "poor"
        recommendations.append(
            "Low cache hit rate — check prompt structure. "
            "Stable system prompts and tool schemas should be prefix-cached."
        )
    elif hit_rate > 0:
        effectiveness = "minimal"
        recommendations.append(
            "Very low cache hits — workload may have unique prompts. "
            "Consider if prefix caching is appropriate for this workload."
        )
    else:
        effectiveness = "disabled_or_no_data"
        recommendations.append(
            "No cache hit data — is prefix caching enabled? "
            "vLLM V1 has zero-overhead prefix caching (always on). "
            "SGLang needs --enable-metrics to expose cache_hit_rate."
        )

    if metrics.engine == "sglang" and hit_rate < 0.5:
        recommendations.append(
            "SGLang RadixAttention typically achieves 85-95% hit rates on coding workloads. "
            "Consider --schedule-policy lpm for longest-prefix-match routing."
        )

    return {
        "cache": {
            "effectiveness": effectiveness,
            "prefix_hit_rate": round(hit_rate, 4),
            "kv_cache_usage": round(kv_usage, 4),
        },
        "recommendations": recommendations,
        "engine": metrics.engine,
        "summary": (f"Cache effectiveness: {effectiveness} (hit rate: {hit_rate:.0%}, KV usage: {kv_usage:.0%})"),
        "confidence": 0.8,
        "evidence": "live_cache_metrics",
    }


def _assess_health(m: NormalizedMetrics) -> dict:
    """Derive health indicators from normalized metrics."""
    issues = []
    status = "healthy"

    # Queue buildup
    if m.requests_waiting > 10:
        issues.append(
            {
                "indicator": "queue_buildup",
                "severity": "warning" if m.requests_waiting < 50 else "critical",
                "detail": f"{m.requests_waiting:.0f} requests waiting in queue",
            }
        )
        status = "degraded"

    # KV cache pressure
    if m.kv_cache_usage > 0.95:
        issues.append(
            {
                "indicator": "kv_cache_critical",
                "severity": "critical",
                "detail": f"KV cache at {m.kv_cache_usage:.0%} — preemptions likely",
            }
        )
        status = "critical"
    elif m.kv_cache_usage > 0.85:
        issues.append(
            {
                "indicator": "kv_cache_high",
                "severity": "warning",
                "detail": f"KV cache at {m.kv_cache_usage:.0%}",
            }
        )
        if status == "healthy":
            status = "degraded"

    # High latency
    if m.ttft_avg_s and m.ttft_avg_s > 5.0:
        issues.append(
            {
                "indicator": "high_ttft",
                "severity": "warning",
                "detail": f"Average TTFT is {m.ttft_avg_s * 1000:.0f}ms",
            }
        )

    if m.itl_avg_s and m.itl_avg_s > 0.1:
        issues.append(
            {
                "indicator": "high_itl",
                "severity": "warning",
                "detail": f"Average ITL is {m.itl_avg_s * 1000:.0f}ms",
            }
        )

    # Low speculation acceptance
    if m.spec_acceptance_rate > 0 and m.spec_acceptance_rate < 0.55:
        issues.append(
            {
                "indicator": "low_spec_acceptance",
                "severity": "warning",
                "detail": (
                    f"Speculative decode acceptance at {m.spec_acceptance_rate:.0%} — "
                    "consider disabling at high concurrency"
                ),
            }
        )

    return {
        "status": status,
        "issues": issues,
        "issue_count": len(issues),
    }


def _build_summary(m: NormalizedMetrics, health: dict) -> str:
    """Build a human-readable summary."""
    parts = [f"{m.engine.upper()} at {m.endpoint}"]
    parts.append(f"Status: {health['status']}")
    parts.append(f"{m.requests_running:.0f} running, {m.requests_waiting:.0f} waiting")
    parts.append(f"KV cache: {m.kv_cache_usage:.0%}")

    if m.ttft_avg_s:
        parts.append(f"TTFT avg: {m.ttft_avg_s * 1000:.0f}ms")
    if m.itl_avg_s:
        parts.append(f"ITL avg: {m.itl_avg_s * 1000:.0f}ms")
    if m.gen_throughput_tps > 0:
        parts.append(f"Throughput: {m.gen_throughput_tps:.0f} tok/s")

    if health["issue_count"] > 0:
        parts.append(f"⚠ {health['issue_count']} issue(s) detected")

    return " | ".join(parts)
