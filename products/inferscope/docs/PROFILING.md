# Runtime Profiling

InferScope ships a Prometheus-first runtime profiling surface for live inference endpoints.

## Public surfaces

- CLI: `inferscope profile-runtime`
- MCP: `tool_profile_runtime`

`inferscope profile` remains the static model-intel command.

## What the runtime profile includes

- normalized runtime metrics
- health summary
- memory pressure analysis
- cache effectiveness analysis
- heuristic workload classification
- audit findings
- grouped bottlenecks
- optional tuning preview
- optional runtime identity and `/v1/models` enrichment
- profiling intent for future `nsys` / `rocprofv3` escalation

## Data flow

1. scrape `/metrics`
2. normalize engine-specific metrics into a shared schema
3. classify workload
4. run deployment checks
5. group findings into bottlenecks
6. preview tuning changes
7. optionally enrich runtime identity from `/v1/models`

The profile report uses the same `MetricSnapshot` schema as benchmark artifacts so live runtime analysis and benchmark telemetry stay compatible.

## CLI examples

Basic profile:

```bash
inferscope profile-runtime http://localhost:8000
```

Profile with hardware hints and tuning preview:

```bash
inferscope profile-runtime http://localhost:8000 \
  --gpu-arch sm_90a \
  --model-name DeepSeek-R1 \
  --quantization fp8 \
  --kv-cache-dtype fp8_e4m3
```

Profile with current scheduler/cache JSON:

```bash
inferscope profile-runtime http://localhost:8000 \
  --current-scheduler '{"batched_token_budget":8192,"decode_priority":0.5}' \
  --current-cache '{"gpu_memory_utilization":0.92,"offload_policy":"cold_only"}'
```

## MCP notes

`tool_profile_runtime` is the MCP-safe wrapper around the same profiling core.

- private IP ranges are blocked by default
- metrics auth is resolved from the MCP payload
- runtime identity enrichment follows the same network policy

## Scope and limits

v1 is intentionally Prometheus-first.

- it does **not** launch `nsys` (NVIDIA) or `rocprofv3` (AMD)
- it does **not** persist runtime profiles to disk by default
- runtime identity enrichment is best-effort, not guaranteed

Future trace and kernel work belongs under `src/inferscope/profiling/`.

## Dynamo / NIXL telemetry

When Dynamo orchestrates a disaggregated deployment, InferScope captures metrics at two levels:

**Worker-level metrics** (scraped from vLLM/SGLang endpoints):
- Standard vLLM/SGLang Prometheus metrics (TTFT, ITL, KV cache utilization, prefix cache hit rate, etc.)
- Each prefill and decode worker is scraped independently
- The decode worker serves as the primary benchmark metrics endpoint

**Orchestration-layer metrics** (from Dynamo Smart Router):
- Router request latency, routing decisions, SLO compliance rates
- NIXL KV transfer throughput and latency
- KV Block Manager utilization and eviction counters

### Dynamo metric prefix families

InferScope groups Dynamo orchestration metrics by prefix family:

| Prefix | Group Key | Description |
|--------|-----------|-------------|
| `dynamo_component_router_` | `router` | Per-request KV-aware routing decisions and latency |
| `dynamo_router_overhead_` | `router_overhead` | Router scheduling and dispatch overhead |
| `dynamo_frontend_worker_` | `frontend_workers` | Per-worker load, queue depth, and health gauges |

These metrics are exposed via the `/metrics` endpoint on the Dynamo Smart Router (typically port 9100) when `DYN_SYSTEM_PORT` is set.

### Engine detection

The telemetry normalizer auto-detects engines from metric text:
- `vllm:` prefix → vLLM worker
- `sglang:` prefix → SGLang worker
- `dynamo_component_router_`, `dynamo_router_overhead_`, or `dynamo_frontend_worker_` → Dynamo orchestration

Note: vLLM detection takes precedence. A combined endpoint exposing both vLLM worker and Dynamo orchestration metrics will be classified as `"vllm"`. Dynamo detection fires only on the dedicated router endpoint.

### Normalized orchestration output

For `engine="dynamo"` scrapes, the normalizer produces an `orchestration` dict grouped by prefix family:

```json
{
  "orchestration": {
    "router": {
      "requests_total": 100.0,
      "latency_seconds_sum": 2.5
    },
    "router_overhead": {
      "scheduling_ms": 0.4
    },
    "frontend_workers": {
      "queue_depth": 3.0,
      "active_connections": 8.0
    }
  }
}
```

This is supplemental telemetry — it does **not** replace the shared worker metrics (TTFT, ITL, KV cache, etc.) which come from the vLLM/SGLang worker endpoints.

## GPU platform notes

- NVIDIA Hopper/Blackwell: primary validated path. Prometheus metrics from vLLM and SGLang are fully normalized.
- AMD MI300X / MI355X: day-one supported. vLLM on ROCm exposes compatible Prometheus metrics. AMD DME telemetry (port 5000) for GPU-level metrics.
