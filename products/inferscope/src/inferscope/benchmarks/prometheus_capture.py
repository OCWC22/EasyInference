"""Prometheus capture helpers for benchmark runs."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from inferscope.benchmarks.experiments import ResolvedMetricCaptureTarget
from inferscope.benchmarks.models import MetricSampleRecord, MetricSnapshot
from inferscope.endpoint_auth import EndpointAuthConfig
from inferscope.telemetry.normalizer import normalize
from inferscope.telemetry.prometheus import MetricSample, scrape_metrics


def _persistable_samples(samples: list[MetricSample]) -> list[MetricSampleRecord]:
    persisted: list[MetricSampleRecord] = []
    for sample in samples:
        if sample.name.endswith("_bucket"):
            continue
        persisted.append(
            MetricSampleRecord(
                name=sample.name,
                labels=dict(sample.labels),
                value=sample.value,
            )
        )
    return persisted


async def capture_endpoint_snapshot(
    endpoint: str,
    allow_private: bool = True,
    *,
    target_name: str = "primary",
    target_role: str = "primary",
    expected_engine: str | None = None,
    metrics_auth: EndpointAuthConfig | None = None,
) -> MetricSnapshot:
    """Capture raw and normalized metrics for one endpoint."""
    scrape = await scrape_metrics(endpoint, allow_private=allow_private, auth=metrics_auth)
    normalized = normalize(scrape)
    error = scrape.error
    if expected_engine and scrape.engine not in {"unknown", expected_engine}:
        mismatch = f"expected engine '{expected_engine}' but scraped '{scrape.engine}'"
        error = f"{error} | {mismatch}" if error else mismatch
    return MetricSnapshot(
        endpoint=scrape.endpoint,
        engine=scrape.engine,
        target_name=target_name,
        target_role=target_role,
        expected_engine=expected_engine,
        scrape_time_ms=scrape.scrape_time_ms,
        error=error,
        raw_metrics=dict(scrape.raw_metrics),
        normalized_metrics=normalized.to_dict(),
        samples=_persistable_samples(scrape.samples),
    )


async def capture_metrics_targets(
    targets: list[ResolvedMetricCaptureTarget],
    allow_private: bool = True,
    *,
    metrics_auth: EndpointAuthConfig | None = None,
    metrics_auth_overrides: Mapping[str, EndpointAuthConfig] | None = None,
) -> list[MetricSnapshot]:
    """Capture all metrics targets concurrently while preserving declared order."""
    tasks = [
        capture_endpoint_snapshot(
            target.endpoint,
            allow_private=allow_private,
            target_name=target.name,
            target_role=target.role,
            expected_engine=target.expected_engine,
            metrics_auth=(metrics_auth_overrides or {}).get(target.name, metrics_auth),
        )
        for target in targets
    ]
    return list(await asyncio.gather(*tasks))
