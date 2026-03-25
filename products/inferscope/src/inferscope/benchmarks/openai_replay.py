"""OpenAI-compatible workload replay runner."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx
import structlog

from inferscope.benchmarks.experiments import BenchmarkRunPlan, build_run_plan
from inferscope.benchmarks.models import (
    BenchmarkArtifact,
    BenchmarkRequestResult,
    BenchmarkSummary,
    MetricSnapshot,
    WorkloadPack,
    WorkloadRequest,
    slugify,
    utc_now_iso,
)
from inferscope.benchmarks.prometheus_capture import capture_metrics_targets
from inferscope.config import settings
from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers, resolve_auth_config, same_origin
from inferscope.logging import get_logger, sanitize_log_text
from inferscope.security import validate_endpoint

log = get_logger(component="benchmarks.openai_replay")


def build_default_artifact_path(artifact: BenchmarkArtifact) -> Path:
    """Build a default artifact path under the benchmark cache directory."""
    root = settings.benchmark_dir
    root.mkdir(parents=True, exist_ok=True)
    return root / artifact.default_filename


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _extract_usage(payload: dict[str, Any]) -> tuple[int | None, int | None, int | None]:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return None, None, None
    prompt_tokens = _coerce_int(usage.get("prompt_tokens"))
    completion_tokens = _coerce_int(usage.get("completion_tokens"))
    total_tokens = _coerce_int(usage.get("total_tokens"))
    return prompt_tokens, completion_tokens, total_tokens


def _event_has_output(payload: dict[str, Any]) -> bool:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False
    first = choices[0]
    if not isinstance(first, dict):
        return False
    delta = first.get("delta")
    if not isinstance(delta, dict):
        return False
    for key, value in delta.items():
        if key == "role":
            continue
        if value not in (None, "", [], {}):
            return True
    return False


def _build_headers(
    request_auth: EndpointAuthConfig | None,
    request: WorkloadRequest,
    extra_headers: dict[str, str] | None,
    *,
    session_header_name: str,
) -> dict[str, str]:
    headers = build_auth_headers(request_auth, include={"Content-Type": "application/json"})
    if request.session_id:
        headers[session_header_name] = request.session_id
    if extra_headers:
        headers.update(extra_headers)
    headers.update(request.headers)
    return headers


def _build_payload(pack: WorkloadPack, request: WorkloadRequest, model: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [message.model_dump(mode="json", exclude_none=True) for message in request.messages],
        "max_tokens": request.max_tokens,
    }
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.tools:
        payload["tools"] = request.tools
    if request.tool_choice is not None:
        payload["tool_choice"] = request.tool_choice
    metadata = dict(request.metadata)
    if request.session_id and "session_id" not in metadata:
        metadata["session_id"] = request.session_id
    if metadata:
        payload["metadata"] = metadata
    payload["stream"] = pack.stream
    if pack.stream:
        payload["stream_options"] = {"include_usage": True}
    payload.update(request.extra_body)
    return payload


async def _run_stream_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[float | None, int | None, int | None, int | None, int]:
    ttft_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    start = time.monotonic()

    async with client.stream("POST", url, headers=headers, json=payload) as response:
        response.raise_for_status()
        saw_stream_event = False
        fallback_lines: list[str] = []
        event_data_lines: list[str] = []

        def process_event(payload_text: str) -> bool:
            nonlocal ttft_ms, prompt_tokens, completion_tokens, total_tokens, saw_stream_event

            if payload_text == "[DONE]":
                return True
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                return False

            if isinstance(event, dict):
                saw_stream_event = True
                if ttft_ms is None and _event_has_output(event):
                    ttft_ms = (time.monotonic() - start) * 1000
                event_prompt, event_completion, event_total = _extract_usage(event)
                if event_prompt is not None:
                    prompt_tokens = event_prompt
                if event_completion is not None:
                    completion_tokens = event_completion
                if event_total is not None:
                    total_tokens = event_total
            return False

        async for raw_line in response.aiter_lines():
            line = raw_line.rstrip("\r")
            if not line:
                if event_data_lines:
                    should_stop = process_event("\n".join(event_data_lines))
                    event_data_lines = []
                    if should_stop:
                        break
                continue

            if line.startswith("data:"):
                event_data_lines.append(line[5:].lstrip())
            else:
                fallback_lines.append(line)

        if event_data_lines:
            process_event("\n".join(event_data_lines))

        if not saw_stream_event and fallback_lines:
            try:
                fallback_payload = json.loads("\n".join(fallback_lines))
            except json.JSONDecodeError:
                fallback_payload = {}
            if isinstance(fallback_payload, dict):
                prompt_tokens, completion_tokens, total_tokens = _extract_usage(fallback_payload)

        return ttft_ms, prompt_tokens, completion_tokens, total_tokens, response.status_code


async def _run_non_stream_request(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
) -> tuple[float | None, int | None, int | None, int | None, int]:
    response = await client.post(url, headers=headers, json=payload)
    response.raise_for_status()
    parsed = response.json()
    if not isinstance(parsed, dict):
        parsed = {}
    prompt_tokens, completion_tokens, total_tokens = _extract_usage(parsed)
    return None, prompt_tokens, completion_tokens, total_tokens, response.status_code


def _session_log_key(session_id: str | None) -> str | None:
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:12]


async def _run_request(
    client: httpx.AsyncClient,
    endpoint: str,
    pack: WorkloadPack,
    model: str,
    request: WorkloadRequest,
    request_auth: EndpointAuthConfig | None,
    extra_headers: dict[str, str] | None,
    request_log: structlog.stdlib.BoundLogger,
    *,
    session_header_name: str,
) -> BenchmarkRequestResult:
    if request.think_time_ms:
        await asyncio.sleep(request.think_time_ms / 1000.0)

    url = f"{endpoint}{pack.endpoint_path}"
    headers = _build_headers(request_auth, request, extra_headers, session_header_name=session_header_name)
    payload = _build_payload(pack, request, model)

    started_at = utc_now_iso()
    started = time.monotonic()
    request_log.info(
        "benchmark_request_started",
        message_count=len(request.messages),
        has_tools=bool(request.tools),
        max_tokens=request.max_tokens,
    )
    try:
        if pack.stream:
            ttft_ms, prompt_tokens, completion_tokens, total_tokens, status_code = await _run_stream_request(
                client, url, headers, payload
            )
        else:
            ttft_ms, prompt_tokens, completion_tokens, total_tokens, status_code = await _run_non_stream_request(
                client, url, headers, payload
            )
        elapsed_ms = (time.monotonic() - started) * 1000
        request_log.info(
            "benchmark_request_completed",
            elapsed_ms=round(elapsed_ms, 2),
            ttft_ms=(round(ttft_ms, 2) if ttft_ms is not None else None),
            status_code=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
        return BenchmarkRequestResult(
            name=request.name,
            session_id=request.session_id,
            status="ok",
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            status_code=status_code,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        elapsed_ms = (time.monotonic() - started) * 1000
        request_log.error(
            "benchmark_request_failed",
            elapsed_ms=round(elapsed_ms, 2),
            error_type=type(exc).__name__,
            error_summary=sanitize_log_text(str(exc)),
        )
        return BenchmarkRequestResult(
            name=request.name,
            session_id=request.session_id,
            status="error",
            started_at=started_at,
            completed_at=utc_now_iso(),
            elapsed_ms=elapsed_ms,
            error=str(exc),
        )


def _build_summary(
    results: list[BenchmarkRequestResult],
    concurrency: int,
    wall_time_ms: float,
    *,
    metrics_targets_total: int = 0,
    metrics_targets_with_errors: int = 0,
    metrics_capture_complete: bool = True,
) -> BenchmarkSummary:
    latencies = [result.elapsed_ms for result in results]
    ttfts = [result.ttft_ms for result in results if result.ttft_ms is not None]
    prompt_tokens = sum(result.prompt_tokens or 0 for result in results)
    completion_tokens = sum(result.completion_tokens or 0 for result in results)
    total_tokens = sum(result.total_tokens or 0 for result in results)
    succeeded = sum(1 for result in results if result.status == "ok")
    failed = sum(1 for result in results if result.status == "error")

    return BenchmarkSummary(
        total_requests=len(results),
        succeeded=succeeded,
        failed=failed,
        concurrency=concurrency,
        wall_time_ms=wall_time_ms,
        latency_avg_ms=(sum(latencies) / len(latencies) if latencies else None),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        latency_p99_ms=_percentile(latencies, 0.99),
        ttft_avg_ms=(sum(ttfts) / len(ttfts) if ttfts else None),
        ttft_p95_ms=_percentile(ttfts, 0.95),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        metrics_targets_total=metrics_targets_total,
        metrics_targets_with_errors=metrics_targets_with_errors,
        metrics_capture_complete=metrics_capture_complete,
    )


def _group_requests(requests: list[WorkloadRequest]) -> list[list[tuple[int, WorkloadRequest]]]:
    grouped: dict[str, list[tuple[int, WorkloadRequest]]] = {}
    order: list[str] = []
    for index, request in enumerate(requests):
        key = request.session_id or f"__request_{index}"
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append((index, request))
    return [grouped[key] for key in order]


def _select_primary_snapshot(snapshots: list[MetricSnapshot]) -> MetricSnapshot | None:
    for snapshot in snapshots:
        if snapshot.target_role == "primary":
            return snapshot
    return snapshots[0] if snapshots else None


def _metrics_capture_status(
    before: list[MetricSnapshot],
    after: list[MetricSnapshot],
    enabled: bool,
) -> tuple[int, int, bool]:
    if not enabled:
        return 0, 0, True
    total = len(before) if before else len(after)
    targets_with_errors = {snapshot.target_name for snapshot in before + after if snapshot.error}
    return total, len(targets_with_errors), not targets_with_errors


async def run_openai_replay(
    workload: WorkloadPack,
    endpoint: str,
    *,
    model: str | None = None,
    metrics_endpoint: str | None = None,
    run_plan: BenchmarkRunPlan | None = None,
    workload_ref: str = "",
    api_key: str | None = None,
    provider: str = "",
    metrics_provider: str = "",
    auth_scheme: str = "",
    auth_header_name: str = "",
    metrics_api_key: str | None = None,
    metrics_auth_scheme: str = "",
    metrics_auth_header_name: str = "",
    metrics_headers: dict[str, str] | None = None,
    concurrency: int | None = None,
    allow_private: bool = True,
    capture_metrics: bool = True,
    extra_headers: dict[str, str] | None = None,
    client: httpx.AsyncClient | None = None,
) -> BenchmarkArtifact:
    """Replay a workload pack against an OpenAI-compatible endpoint."""
    validated_endpoint = validate_endpoint(endpoint, allow_private=allow_private)
    validated_metrics_endpoint = validate_endpoint(metrics_endpoint or validated_endpoint, allow_private=allow_private)

    resolved_plan = (
        run_plan.model_copy(deep=True)
        if run_plan
        else build_run_plan(
            workload,
            validated_endpoint,
            workload_ref=workload_ref or workload.name,
            model=model,
            concurrency=concurrency,
            metrics_endpoint=validated_metrics_endpoint,
        )
    )
    if run_plan is not None:
        resolved_plan.request_endpoint = validated_endpoint

    if resolved_plan.topology.session_routing in {"sticky", "hash"} and any(
        request.session_id is None for request in workload.requests
    ):
        raise ValueError(
            "All workload requests must include session_id when session routing is enabled for the run plan"
        )

    request_auth = resolve_auth_config(
        api_key,
        provider=provider,
        auth_scheme=auth_scheme,
        auth_header_name=auth_header_name,
        default_scheme="bearer",
    )
    reuse_request_auth_for_metrics = (
        metrics_api_key is None and not metrics_headers and same_origin(validated_endpoint, validated_metrics_endpoint)
    )
    metrics_auth = resolve_auth_config(
        metrics_api_key if metrics_api_key is not None else (api_key if reuse_request_auth_for_metrics else None),
        provider=metrics_provider or provider,
        auth_scheme=metrics_auth_scheme,
        auth_header_name=metrics_auth_header_name,
        headers=metrics_headers,
        default_scheme="bearer",
    )

    benchmark_id = f"{slugify(workload.name)}-{uuid4().hex[:12]}"
    run_log = log.bind(
        benchmark_id=benchmark_id,
        endpoint=validated_endpoint,
        workload=workload.name,
        workload_class=workload.workload_class,
        concurrency=resolved_plan.concurrency,
        model=resolved_plan.model,
        topology_mode=resolved_plan.topology.mode,
        cache_strategy=resolved_plan.cache.strategy,
    )
    run_log.info(
        "benchmark_started",
        total_requests=len(workload.requests),
        capture_metrics=capture_metrics,
        metrics_targets=len(resolved_plan.metrics_targets),
        endpoint_path=workload.endpoint_path,
        stream=workload.stream,
    )

    metrics_before_targets: list[MetricSnapshot] = []
    if capture_metrics:
        metrics_before_targets = await capture_metrics_targets(
            resolved_plan.metrics_targets,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
        )
        run_log.info(
            "benchmark_metrics_captured_before",
            captured_targets=len(metrics_before_targets),
            targets_with_errors=sum(1 for snapshot in metrics_before_targets if snapshot.error),
        )

    started_at = utc_now_iso()
    started = time.monotonic()

    created_client = client is None
    active_client = client or httpx.AsyncClient(timeout=60.0)
    semaphore = asyncio.Semaphore(resolved_plan.concurrency)
    request_groups = _group_requests(workload.requests)

    async def run_group(group: list[tuple[int, WorkloadRequest]]) -> list[tuple[int, BenchmarkRequestResult]]:
        async with semaphore:
            group_results: list[tuple[int, BenchmarkRequestResult]] = []
            for index, request in group:
                request_log = run_log.bind(
                    request_name=request.name,
                    session_present=request.session_id is not None,
                    session_key=_session_log_key(request.session_id),
                )
                result = await _run_request(
                    active_client,
                    validated_endpoint,
                    workload,
                    resolved_plan.model,
                    request,
                    request_auth,
                    extra_headers,
                    request_log,
                    session_header_name=resolved_plan.topology.session_header_name,
                )
                group_results.append((index, result))
            return group_results

    try:
        grouped_results = await asyncio.gather(*(run_group(group) for group in request_groups))
    finally:
        if created_client:
            await active_client.aclose()

    indexed_results = [item for group in grouped_results for item in group]
    indexed_results.sort(key=lambda item: item[0])
    results = [result for _, result in indexed_results]

    completed_at = utc_now_iso()
    wall_time_ms = (time.monotonic() - started) * 1000

    metrics_after_targets: list[MetricSnapshot] = []
    if capture_metrics:
        metrics_after_targets = await capture_metrics_targets(
            resolved_plan.metrics_targets,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
        )
        run_log.info(
            "benchmark_metrics_captured_after",
            captured_targets=len(metrics_after_targets),
            targets_with_errors=sum(1 for snapshot in metrics_after_targets if snapshot.error),
        )

    metrics_targets_total, metrics_targets_with_errors, metrics_capture_complete = _metrics_capture_status(
        metrics_before_targets,
        metrics_after_targets,
        capture_metrics,
    )
    summary = _build_summary(
        results,
        resolved_plan.concurrency,
        wall_time_ms,
        metrics_targets_total=metrics_targets_total,
        metrics_targets_with_errors=metrics_targets_with_errors,
        metrics_capture_complete=metrics_capture_complete,
    )

    primary_before = _select_primary_snapshot(metrics_before_targets)
    primary_after = _select_primary_snapshot(metrics_after_targets)
    artifact = BenchmarkArtifact(
        benchmark_id=benchmark_id,
        pack_name=workload.name,
        workload_class=workload.workload_class,
        endpoint=validated_endpoint,
        metrics_endpoint=(
            primary_after.endpoint if primary_after else primary_before.endpoint if primary_before else None
        ),
        model=resolved_plan.model,
        concurrency=resolved_plan.concurrency,
        started_at=started_at,
        completed_at=completed_at,
        run_plan=resolved_plan.model_dump(mode="json"),
        metrics_before=primary_before,
        metrics_after=primary_after,
        metrics_before_targets=metrics_before_targets,
        metrics_after_targets=metrics_after_targets,
        results=results,
        summary=summary,
    )
    run_log.info(
        "benchmark_completed",
        total_requests=summary.total_requests,
        succeeded=summary.succeeded,
        failed=summary.failed,
        wall_time_ms=round(summary.wall_time_ms, 2),
        metrics_targets=summary.metrics_targets_total,
        metrics_target_errors=summary.metrics_targets_with_errors,
        metrics_capture_complete=summary.metrics_capture_complete,
        latency_p95_ms=(round(summary.latency_p95_ms, 2) if summary.latency_p95_ms is not None else None),
        ttft_p95_ms=(round(summary.ttft_p95_ms, 2) if summary.ttft_p95_ms is not None else None),
    )
    return artifact
