"""Tests for Dynamo engine detection and orchestration metrics normalization."""

from __future__ import annotations

from inferscope.telemetry.prometheus import (
    MetricSample,
    ScrapeResult,
    _parse_prometheus_labels,
    detect_engine_from_metrics,
    parse_prometheus_text,
)
from inferscope.telemetry.normalizer import normalize


# ---------------------------------------------------------------------------
# detect_engine_from_metrics
# ---------------------------------------------------------------------------


def test_detect_engine_dynamo_from_router_metrics() -> None:
    text = (
        "# HELP dynamo_component_router_requests_total Total routed requests\n"
        "dynamo_component_router_requests_total 42\n"
    )
    assert detect_engine_from_metrics(text) == "dynamo"


def test_detect_engine_dynamo_from_overhead_metrics() -> None:
    text = "dynamo_router_overhead_seconds_sum 0.5\n"
    assert detect_engine_from_metrics(text) == "dynamo"


def test_detect_engine_dynamo_from_frontend_worker_metrics() -> None:
    text = "dynamo_frontend_worker_queue_depth 3\n"
    assert detect_engine_from_metrics(text) == "dynamo"


def test_detect_engine_vllm_takes_precedence_over_dynamo() -> None:
    """When vLLM metrics are present alongside Dynamo, detect vLLM (worker endpoint)."""
    text = (
        "vllm:num_requests_running 5\n"
        "dynamo_component_router_requests_total 42\n"
    )
    assert detect_engine_from_metrics(text) == "vllm"


def test_detect_engine_unknown_when_no_prefixes() -> None:
    text = "some_random_metric 1.0\n"
    assert detect_engine_from_metrics(text) == "unknown"


# ---------------------------------------------------------------------------
# _parse_prometheus_labels — quote-aware label parsing
# ---------------------------------------------------------------------------


def test_parse_labels_simple() -> None:
    result = _parse_prometheus_labels('worker="prefill-0",method="POST"')
    assert result == {"worker": "prefill-0", "method": "POST"}


def test_parse_labels_with_commas_in_value() -> None:
    result = _parse_prometheus_labels('name="hello,world",code="200"')
    assert result == {"name": "hello,world", "code": "200"}


def test_parse_labels_with_escaped_quotes() -> None:
    result = _parse_prometheus_labels('path="/api/v1/\\"test\\"",method="GET"')
    assert result == {"path": '/api/v1/"test"', "method": "GET"}


def test_parse_labels_empty_string() -> None:
    assert _parse_prometheus_labels("") == {}


def test_parse_labels_unquoted_value() -> None:
    result = _parse_prometheus_labels("code=200")
    assert result == {"code": "200"}


def test_parse_prometheus_text_with_labels() -> None:
    text = (
        '# HELP test_metric A test\n'
        'test_metric{worker="prefill-0",method="POST"} 42\n'
        'test_metric{worker="decode-0",method="POST"} 58\n'
    )
    samples = parse_prometheus_text(text)
    labeled = [s for s in samples if s.name == "test_metric"]
    assert len(labeled) == 2
    assert labeled[0].labels["worker"] in ("prefill-0", "decode-0")


# ---------------------------------------------------------------------------
# normalize — Dynamo orchestration grouping
# ---------------------------------------------------------------------------


def _make_dynamo_scrape() -> ScrapeResult:
    return ScrapeResult(
        endpoint="http://localhost:9100/metrics",
        engine="dynamo",
        raw_metrics={
            "dynamo_component_router_requests_total": 100.0,
            "dynamo_component_router_latency_seconds_sum": 2.5,
            "dynamo_router_overhead_scheduling_ms": 0.4,
            "dynamo_frontend_worker_queue_depth": 3.0,
            "dynamo_frontend_worker_active_connections": 8.0,
        },
        samples=[
            MetricSample(name="dynamo_component_router_requests_total", labels={}, value=100.0),
            MetricSample(name="dynamo_component_router_latency_seconds_sum", labels={}, value=2.5),
            MetricSample(name="dynamo_router_overhead_scheduling_ms", labels={}, value=0.4),
            MetricSample(name="dynamo_frontend_worker_queue_depth", labels={}, value=3.0),
            MetricSample(name="dynamo_frontend_worker_active_connections", labels={}, value=8.0),
        ],
        scrape_time_ms=5.0,
    )


def test_normalize_dynamo_populates_orchestration_groups() -> None:
    scrape = _make_dynamo_scrape()
    result = normalize(scrape)

    assert result.orchestration is not None
    assert "router" in result.orchestration
    assert "router_overhead" in result.orchestration
    assert "frontend_workers" in result.orchestration


def test_normalize_dynamo_router_group_strips_prefix() -> None:
    scrape = _make_dynamo_scrape()
    result = normalize(scrape)

    router = result.orchestration["router"]
    assert "requests_total" in router
    assert router["requests_total"] == 100.0
    assert "latency_seconds_sum" in router
    assert router["latency_seconds_sum"] == 2.5


def test_normalize_dynamo_frontend_workers_group() -> None:
    scrape = _make_dynamo_scrape()
    result = normalize(scrape)

    workers = result.orchestration["frontend_workers"]
    assert workers["queue_depth"] == 3.0
    assert workers["active_connections"] == 8.0


def test_normalize_dynamo_serializes_orchestration_in_dict() -> None:
    scrape = _make_dynamo_scrape()
    result = normalize(scrape)
    d = result.to_dict()

    assert "orchestration" in d
    assert d["orchestration"]["router"]["requests_total"] == 100.0


def test_normalize_dynamo_empty_raw_metrics_no_orchestration() -> None:
    scrape = ScrapeResult(
        endpoint="http://localhost:9100/metrics",
        engine="dynamo",
        raw_metrics={},
        samples=[],
        scrape_time_ms=1.0,
    )
    result = normalize(scrape)
    assert result.orchestration is None


def test_normalize_dynamo_labeled_samples_preserve_labels() -> None:
    """When Dynamo metrics have labels, normalization preserves them as list-of-dicts."""
    scrape = ScrapeResult(
        endpoint="http://localhost:9100/metrics",
        engine="dynamo",
        raw_metrics={},
        samples=[
            MetricSample(
                name="dynamo_component_router_requests_total",
                labels={"worker": "prefill-0", "method": "POST"},
                value=60.0,
            ),
            MetricSample(
                name="dynamo_component_router_requests_total",
                labels={"worker": "decode-0", "method": "POST"},
                value=40.0,
            ),
            MetricSample(
                name="dynamo_component_router_latency_seconds_sum",
                labels={},
                value=2.5,
            ),
        ],
        scrape_time_ms=5.0,
    )
    result = normalize(scrape)
    assert result.orchestration is not None

    router = result.orchestration["router"]
    # Labeled metric becomes a list of {labels, value} dicts
    assert isinstance(router["requests_total"], list)
    assert len(router["requests_total"]) == 2
    assert router["requests_total"][0]["labels"]["worker"] in ("prefill-0", "decode-0")
    assert router["requests_total"][0]["value"] in (60.0, 40.0)

    # Unlabeled metric remains a scalar
    assert router["latency_seconds_sum"] == 2.5


def test_normalize_vllm_has_no_orchestration() -> None:
    scrape = ScrapeResult(
        endpoint="http://localhost:8000/metrics",
        engine="vllm",
        raw_metrics={"vllm:num_requests_running": 2.0},
        samples=[MetricSample(name="vllm:num_requests_running", labels={}, value=2.0)],
        scrape_time_ms=3.0,
    )
    result = normalize(scrape)
    assert result.orchestration is None
