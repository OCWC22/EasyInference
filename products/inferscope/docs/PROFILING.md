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

## GPU platform notes

- NVIDIA Hopper/Blackwell: primary validated path. Prometheus metrics from vLLM and SGLang are fully normalized.
- AMD MI300X / MI355X: day-one supported. vLLM on ROCm exposes compatible Prometheus metrics. AMD DME telemetry (port 5000) for GPU-level metrics.
