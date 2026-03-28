"""Cross-engine metric normalization.

Converts vLLM, SGLang, ATOM, and Dynamo metrics into a common InferScope format
so audit checks and diagnostics work regardless of engine.
"""

from __future__ import annotations

from dataclasses import dataclass

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

    # Dynamo reliability/observability signals
    request_migrations_total: float = 0.0
    disconnected_clients: float = 0.0
    kv_active_blocks: float = 0.0
    kv_total_blocks: float = 0.0

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
            "reliability": {
                "request_migrations_total": self.request_migrations_total,
                "disconnected_clients": self.disconnected_clients,
                "kv_active_blocks": self.kv_active_blocks,
                "kv_total_blocks": self.kv_total_blocks,
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
        # vLLM v0.18 renamed gpu_cache_usage_perc -> kv_cache_usage_perc
        # Use `is not None` to avoid falsy-zero bug (0.0 is a valid value)
        kv_new = scrape.get("vllm:kv_cache_usage_perc")
        m.kv_cache_usage = kv_new if kv_new is not None else scrape.get("vllm:gpu_cache_usage_perc")
        # vLLM v0.18 replaced prefix_cache_hit_rate gauge with counters
        m.prefix_cache_hit_rate = scrape.get("vllm:gpu_prefix_cache_hit_rate")
        if m.prefix_cache_hit_rate is None:
            hits = scrape.get("vllm:prefix_cache_hits_total")
            queries = scrape.get("vllm:prefix_cache_queries_total")
            if hits is not None and queries is not None and queries > 0:
                m.prefix_cache_hit_rate = hits / queries
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
        m.requests_running = scrape.get("dynamo_frontend_inflight_requests") or scrape.get(
            "dynamo_component_inflight_requests"
        )
        m.requests_waiting = scrape.get("dynamo_frontend_queued_requests")
        # Dynamo passes through vLLM KV cache metrics — no dynamo-specific KV metric exists
        kv_new = scrape.get("vllm:kv_cache_usage_perc")
        m.kv_cache_usage = kv_new if kv_new is not None else scrape.get("vllm:gpu_cache_usage_perc")
        # Prefix cache: try Dynamo router KV hit rate, fall back to vLLM counters
        m.prefix_cache_hit_rate = scrape.get("dynamo_component_router_kv_hit_rate")
        if m.prefix_cache_hit_rate is None:
            hits = scrape.get("vllm:prefix_cache_hits_total")
            queries = scrape.get("vllm:prefix_cache_queries_total")
            if hits is not None and queries is not None and queries > 0:
                m.prefix_cache_hit_rate = hits / queries
        m.prompt_tokens_total = scrape.get("vllm:prompt_tokens_total")
        m.generation_tokens_total = scrape.get("dynamo_frontend_output_tokens_total") or scrape.get(
            "vllm:generation_tokens_total"
        )
        m.request_success_total = scrape.get("dynamo_frontend_requests_total") or scrape.get(
            "dynamo_component_requests_total"
        )
        m.ttft_avg_s = scrape.get_histogram_avg(
            "dynamo_frontend_time_to_first_token_seconds"
        ) or scrape.get_histogram_avg("vllm:time_to_first_token_seconds")
        m.itl_avg_s = scrape.get_histogram_avg(
            "dynamo_frontend_inter_token_latency_seconds"
        ) or scrape.get_histogram_avg("vllm:time_per_output_token_seconds")
        m.e2e_avg_s = scrape.get_histogram_avg("dynamo_frontend_request_duration_seconds") or scrape.get_histogram_avg(
            "dynamo_component_request_duration_seconds"
        )
        m.request_migrations_total = scrape.get("dynamo_frontend_model_migration_total")
        m.disconnected_clients = scrape.get("dynamo_frontend_disconnected_clients")
        m.kv_active_blocks = scrape.get("dynamo_component_kvstats_active_blocks")
        m.kv_total_blocks = scrape.get("dynamo_component_kvstats_total_blocks")

    return m
