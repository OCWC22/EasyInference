"""Cross-engine metric normalization.

Converts vLLM, SGLang, ATOM, and Dynamo metrics into a common InferScope format
so audit checks and diagnostics work regardless of engine.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any

from inferscope.telemetry.prometheus import ScrapeResult


@dataclass
class NormalizedMetrics:
    """Engine-agnostic metric snapshot for InferScope analysis."""

    engine: str
    endpoint: str

    # Request state
    requests_running: float = 0.0
    requests_waiting: float = 0.0
    requests_swapped: float = 0.0  # vLLM only

    # Cache
    kv_cache_usage: float = 0.0  # 0-1
    prefix_cache_hit_rate: float = 0.0  # 0-1
    cpu_cache_usage: float = 0.0  # 0-1

    # Throughput (counters — need rate() for per-second)
    prompt_tokens_total: float = 0.0
    generation_tokens_total: float = 0.0
    preemptions_total: float = 0.0
    request_success_total: float = 0.0

    # Latency (histogram averages in seconds)
    ttft_avg_s: float | None = None  # Time to first token
    itl_avg_s: float | None = None  # Inter-token latency
    e2e_avg_s: float | None = None  # End-to-end latency
    queue_time_avg_s: float | None = None

    # Speculative decoding
    spec_acceptance_rate: float = 0.0

    # Generation throughput (gauge, tokens/sec — SGLang only)
    gen_throughput_tps: float = 0.0

    # Dynamo orchestration metrics (populated only for Dynamo router targets).
    # Values are either scalar floats (unlabeled) or lists of {"labels": ..., "value": ...}
    # dicts (labeled, preserving per-worker/per-route dimensions).
    orchestration: dict[str, dict[str, Any]] | None = None

    # Scrape metadata
    scrape_time_ms: float = 0.0
    scrape_error: str = ""

    def to_dict(self) -> dict:
        return {
            "engine": self.engine,
            "endpoint": self.endpoint,
            "request_state": {
                "running": self.requests_running,
                "waiting": self.requests_waiting,
                "swapped": self.requests_swapped,
            },
            "cache": {
                "kv_usage": round(self.kv_cache_usage, 4),
                "prefix_hit_rate": round(self.prefix_cache_hit_rate, 4),
                "cpu_usage": round(self.cpu_cache_usage, 4),
            },
            "throughput": {
                "prompt_tokens_total": self.prompt_tokens_total,
                "generation_tokens_total": self.generation_tokens_total,
                "preemptions_total": self.preemptions_total,
                "request_success_total": self.request_success_total,
                "gen_throughput_tps": round(self.gen_throughput_tps, 1),
            },
            "latency": {
                "ttft_avg_ms": round(self.ttft_avg_s * 1000, 1) if self.ttft_avg_s else None,
                "itl_avg_ms": round(self.itl_avg_s * 1000, 1) if self.itl_avg_s else None,
                "e2e_avg_ms": round(self.e2e_avg_s * 1000, 1) if self.e2e_avg_s else None,
                "queue_time_avg_ms": (round(self.queue_time_avg_s * 1000, 1) if self.queue_time_avg_s else None),
            },
            "speculation": {
                "acceptance_rate": round(self.spec_acceptance_rate, 3),
            },
            "scrape": {
                "time_ms": round(self.scrape_time_ms, 1),
                "error": self.scrape_error,
            },
            **({
                "orchestration": self.orchestration,
            } if self.orchestration else {}),
        }


def normalize(scrape: ScrapeResult) -> NormalizedMetrics:
    """Convert engine-specific ScrapeResult into NormalizedMetrics."""
    m = NormalizedMetrics(
        engine=scrape.engine,
        endpoint=scrape.endpoint,
        scrape_time_ms=scrape.scrape_time_ms,
        scrape_error=scrape.error,
    )

    if scrape.error:
        return m

    if scrape.engine == "vllm":
        m.requests_running = scrape.get("vllm:num_requests_running")
        m.requests_waiting = scrape.get("vllm:num_requests_waiting")
        m.requests_swapped = scrape.get("vllm:num_requests_swapped")
        m.kv_cache_usage = scrape.get("vllm:gpu_cache_usage_perc")
        m.prefix_cache_hit_rate = scrape.get("vllm:gpu_prefix_cache_hit_rate")
        m.cpu_cache_usage = scrape.get("vllm:cpu_cache_usage_perc")
        m.prompt_tokens_total = scrape.get("vllm:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("vllm:generation_tokens_total")
        m.preemptions_total = scrape.get("vllm:num_preemptions_total")
        m.request_success_total = scrape.get("vllm:request_success_total")
        m.spec_acceptance_rate = scrape.get("vllm:spec_decode_draft_acceptance_rate")
        m.ttft_avg_s = scrape.get_histogram_avg("vllm:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("vllm:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("vllm:e2e_request_latency_seconds")
        m.queue_time_avg_s = scrape.get_histogram_avg("vllm:request_queue_time_seconds")

    elif scrape.engine == "sglang":
        m.requests_running = scrape.get("sglang:num_running_reqs")
        m.requests_waiting = scrape.get("sglang:num_queue_reqs")
        m.kv_cache_usage = scrape.get("sglang:token_usage")
        m.prefix_cache_hit_rate = scrape.get("sglang:cache_hit_rate")
        m.prompt_tokens_total = scrape.get("sglang:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("sglang:generation_tokens_total")
        m.gen_throughput_tps = scrape.get("sglang:gen_throughput")
        m.ttft_avg_s = scrape.get_histogram_avg("sglang:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("sglang:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("sglang:e2e_request_latency_seconds")

    elif scrape.engine == "atom":
        # ATOM follows vLLM schema with atom: prefix
        m.requests_running = scrape.get("atom:num_requests_running")
        m.requests_waiting = scrape.get("atom:num_requests_waiting")
        m.kv_cache_usage = scrape.get("atom:kv_cache_usage_perc")
        m.ttft_avg_s = scrape.get_histogram_avg("atom:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg("atom:inter_token_latency_seconds")

    elif scrape.engine == "dynamo":
        # Dynamo router/orchestration metrics — grouped by prefix family.
        # Uses samples (not raw_metrics) to preserve per-worker/per-route label dimensions.
        _dynamo_prefixes = {
            "router": "dynamo_component_router_",
            "router_overhead": "dynamo_router_overhead_",
            "frontend_workers": "dynamo_frontend_worker_",
        }
        orchestration: dict[str, dict[str, Any]] = {}
        for group_name, prefix in _dynamo_prefixes.items():
            # Collect all samples in this family, keyed by stripped metric name
            metric_entries: dict[str, list[tuple[dict[str, str], float]]] = {}
            for sample in scrape.samples:
                if sample.name.startswith(prefix):
                    key = sample.name[len(prefix):]
                    metric_entries.setdefault(key, []).append((dict(sample.labels), sample.value))
            if not metric_entries:
                continue
            group: dict[str, Any] = {}
            for key, entries in sorted(metric_entries.items()):
                if len(entries) == 1 and not entries[0][0]:
                    # Single unlabeled sample — store as scalar for simplicity
                    group[key] = entries[0][1]
                else:
                    # Multiple samples or labeled — preserve full label dimensions
                    group[key] = [
                        {"labels": labels, "value": value}
                        for labels, value in sorted(entries, key=lambda e: sorted(e[0].items()))
                    ]
            orchestration[group_name] = group
        if orchestration:
            m.orchestration = orchestration

    return m
