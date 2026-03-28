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

    # KV disaggregation metrics
    kvbm_offload_d2h: float = 0.0
    kvbm_onboard_h2d: float = 0.0
    nixl_transfer_latency_s: float | None = None
    nixl_transfer_bytes: float = 0.0
    nixl_transfer_failures: float = 0.0

    # Tiering metrics
    kvbm_host_hit_rate: float = 0.0
    kvbm_disk_hit_rate: float = 0.0
    grove_tier_gpu_pct: float = 0.0
    grove_tier_cpu_pct: float = 0.0
    grove_tier_ssd_pct: float = 0.0
    grove_evictions: float = 0.0

    # LMCache metrics
    lmcache_hit_rate: float = 0.0
    lmcache_retrieve_speed_tps: float = 0.0

    # SLO violation counters
    slo_ttft_violations: float = 0.0
    slo_itl_violations: float = 0.0

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
            "disaggregation": {
                "kvbm_offload_d2h": self.kvbm_offload_d2h,
                "kvbm_onboard_h2d": self.kvbm_onboard_h2d,
                "nixl_transfer_latency_ms": round(self.nixl_transfer_latency_s * 1000, 2) if self.nixl_transfer_latency_s else None,
                "nixl_transfer_bytes": self.nixl_transfer_bytes,
                "nixl_transfer_failures": self.nixl_transfer_failures,
            },
            "tiering": {
                "kvbm_host_hit_rate": self.kvbm_host_hit_rate,
                "kvbm_disk_hit_rate": self.kvbm_disk_hit_rate,
                "grove_gpu_pct": self.grove_tier_gpu_pct,
                "grove_cpu_pct": self.grove_tier_cpu_pct,
                "grove_ssd_pct": self.grove_tier_ssd_pct,
                "grove_evictions": self.grove_evictions,
            },
            "lmcache": {
                "hit_rate": self.lmcache_hit_rate,
                "retrieve_speed_tps": self.lmcache_retrieve_speed_tps,
            },
            "slo": {
                "ttft_violations": self.slo_ttft_violations,
                "itl_violations": self.slo_itl_violations,
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
        m.requests_running = scrape.get("dynamo_frontend_inflight_requests") or scrape.get(
            "dynamo_component_inflight_requests"
        )
        m.requests_waiting = scrape.get("dynamo_frontend_queued_requests")
        m.kv_cache_usage = scrape.get("dynamo_component_kvstats_gpu_cache_usage_percent") or scrape.get(
            "vllm:gpu_cache_usage_perc"
        )
        m.prefix_cache_hit_rate = scrape.get("dynamo_component_kvstats_gpu_prefix_cache_hit_rate") or scrape.get(
            "vllm:gpu_prefix_cache_hit_rate"
        )
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
        # Dynamo KV disaggregation
        m.kvbm_offload_d2h = scrape.get("kvbm_offload_blocks_d2h")
        m.kvbm_onboard_h2d = scrape.get("kvbm_onboard_blocks_h2d")
        m.nixl_transfer_latency_s = scrape.get_histogram_avg("dynamo_nixl_transfer_latency_seconds")
        m.nixl_transfer_bytes = scrape.get("dynamo_nixl_transfer_bytes_total")
        m.nixl_transfer_failures = scrape.get("dynamo_nixl_transfer_failures_total")
        # Tiering
        m.kvbm_host_hit_rate = scrape.get("kvbm_host_cache_hit_rate")
        m.kvbm_disk_hit_rate = scrape.get("kvbm_disk_cache_hit_rate")
        m.grove_tier_gpu_pct = scrape.get("dynamo_grove_tier_gpu_usage_percent")
        m.grove_tier_cpu_pct = scrape.get("dynamo_grove_tier_cpu_usage_percent")
        m.grove_tier_ssd_pct = scrape.get("dynamo_grove_tier_ssd_usage_percent")
        m.grove_evictions = scrape.get("dynamo_grove_evictions_total")
        # LMCache
        m.lmcache_hit_rate = scrape.get("dynamo_lmcache_hit_rate")
        lmc_speed_sum = scrape.get("lmcache:retrieve_speed_sum")
        lmc_speed_count = scrape.get("lmcache:retrieve_speed_count")
        if lmc_speed_count > 0:
            m.lmcache_retrieve_speed_tps = lmc_speed_sum / lmc_speed_count
        # SLO
        m.slo_ttft_violations = scrape.get("dynamo_slo_ttft_violations_total")
        m.slo_itl_violations = scrape.get("dynamo_slo_itl_violations_total")

    return m
