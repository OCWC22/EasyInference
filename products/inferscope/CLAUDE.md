# CLAUDE.md — InferScope Implementation Guide for Claude Code

This file covers low-level implementation decisions for `products/inferscope/`. The root CLAUDE.md covers strategic context.

## SkyPilot Integration Architecture

### Where SkyPilot fits (and doesn't)

```
┌─────────────────────────────────────────────────────┐
│  SkyPilot (external orchestration)                   │
│  - Provisions GPU instances across clouds            │
│  - Manages spot preemption and failover              │
│  - Pins environment (vLLM version, CUDA, model)      │
│  - Uploads artifacts to cloud storage                │
│                                                       │
│  sky launch inferscope-bench.yaml                    │
│       │                                               │
│       ▼                                               │
│  ┌─────────────────────────────────────────────┐     │
│  │  On the provisioned GPU instance:            │     │
│  │                                               │     │
│  │  1. Start vLLM/SGLang (SkyPilot setup task)  │     │
│  │  2. Wait for /health to return 200            │     │
│  │  3. Run: inferscope benchmark \               │     │
│  │       --dataset sharegpt \                    │     │
│  │       http://localhost:8000                    │     │
│  │  4. Upload artifact to S3/GCS                 │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

SkyPilot is the **outer loop**. InferScope is the **inner loop**. They communicate through:
- CLI invocation (SkyPilot runs `inferscope benchmark ...`)
- Artifact files (InferScope writes JSON, SkyPilot uploads to storage)
- Exit codes (InferScope returns 0 on success, non-zero on failure)

### Files to create for SkyPilot integration

Location: `products/inferscope/skypilot/` (new directory)

```
skypilot/
├── README.md                          # How to use these task YAMLs
├── tasks/
│   ├── single-gpu-vllm.yaml          # vLLM on 1xH100, ShareGPT benchmark
│   ├── single-gpu-sglang.yaml        # SGLang on 1xH100, ShareGPT benchmark
│   ├── multi-gpu-vllm.yaml           # vLLM on 4xH100, large model benchmark
│   ├── sweep-gpu-generations.yaml    # Same model across H100/H200/B200
│   └── disagg-prefill-decode.yaml    # 2+ GPU disaggregated serving benchmark
├── scripts/
│   ├── setup-vllm.sh                 # Install vLLM, download model, start server
│   ├── setup-sglang.sh               # Same for SGLang
│   ├── wait-for-health.sh            # Poll /health until ready
│   ├── run-benchmark.sh              # Run InferScope, handle errors
│   └── upload-artifact.sh            # Upload results to S3/GCS
└── envs/
    ├── vllm-h100.env                 # Pinned versions for H100 + vLLM
    ├── vllm-b200.env                 # Pinned versions for B200 + vLLM
    └── sglang-h100.env               # Pinned versions for H100 + SGLang
```

### SkyPilot task YAML structure (example)

```yaml
# single-gpu-vllm.yaml
name: inferscope-vllm-sharegpt

resources:
  accelerators: {H100:1, H200:1, B200:1}  # SkyPilot picks cheapest
  disk_size: 256
  use_spot: true

setup: |
  # Pin exact versions for reproducibility
  pip install vllm==0.8.5 inferscope==0.1.0
  # Download model weights
  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct

run: |
  # Start vLLM in background
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --quantization fp8 --host 0.0.0.0 --port 8000 &
  VLLM_PID=$!

  # Wait for readiness
  for i in $(seq 1 120); do
    curl -sf http://localhost:8000/health && break
    sleep 2
  done

  # Run InferScope benchmark
  inferscope benchmark \
    --dataset sharegpt \
    --dataset-sample-size 256 \
    --model meta-llama/Llama-3.3-70B-Instruct \
    --gpu $(sky accelerator) \
    http://localhost:8000

  # Upload artifact
  ARTIFACT=$(ls -t ~/.inferscope/benchmarks/*.json | head -1)
  aws s3 cp "$ARTIFACT" s3://inferscope-results/$(date +%Y%m%d)/

  kill $VLLM_PID
```

### Implementation constraints for SkyPilot compatibility

When modifying InferScope code, ensure:

1. **Artifact filenames must be deterministic and unique** — include model name, GPU type, timestamp, and benchmark ID. Current `default_filename` property does this.

2. **Exit codes must be meaningful** — `inferscope benchmark` should return:
   - 0: success
   - 1: benchmark completed but zero successful requests
   - 2: configuration error (bad model, unsupported GPU)
   - 3: endpoint unreachable

3. **No interactive prompts** — everything must work in non-interactive mode for SkyPilot task execution.

4. **Artifact must be self-contained** — all metadata (model, GPU, engine, vLLM version, concurrency, workload reference) must be inside the JSON artifact, not inferred from context. Current `BenchmarkArtifact` + `run_plan` achieve this.

5. **GPU detection should be automatic** — when `--gpu` is not specified, InferScope should detect the GPU via `nvidia-smi` or CUDA device properties and populate the artifact. Currently missing — `--gpu` is manual.

## Benchmark Runtime: Implementation Details

### The runtime is the product — guard these invariants

`benchmarks/runtime.py` is the most important file. Key invariants:

1. **Session ordering**: Requests with the same `session_id` execute sequentially within a session, different sessions execute concurrently. This is how real agentic workloads behave. Never break this.

2. **Arrival scheduling**: `_arrival_offsets_ms()` generates deterministic inter-arrival times using seeded RNG. Poisson (default for rate-limited), Gamma (bursty), or immediate (blast). The seed must be in the artifact for reproducibility.

3. **TTFT measurement**: Measured as `time.monotonic()` delta from request start to first SSE chunk with non-empty content. This is operator-grade, not token-precise. The runtime warns about this — don't remove the warning.

4. **Goodput SLO**: `_request_slo()` uses `actual_context_tokens` (real payload size) for threshold bucketing, NOT `target_context_tokens` (planning/template values). This was a deliberate fix — don't revert it.

5. **Tool-call validation**: `_looks_like_tool_call()` uses regex, not JSON parsing. This is intentional — models produce malformed tool calls, and we want to measure parse success rate, not enforce correctness.

### Core types and contracts

From `benchmarks/models.py` — these are the stable API:

- **`WorkloadPack`** — the standard benchmark workload container
  - `requests` must be non-empty
  - `endpoint_path` must start with `/`
  - Honesty fields: `metadata["target_context_tokens"]` (intended), `actual_context_tokens` (real payload)
  - Legacy read-compat: `metadata["approx_context_tokens"]` — do not use for new code
  - `hydration_mode` derived from tags: `hydration:template`, `hydration:synthetic`, `hydration:hydrated`
- **`WorkloadRequest`** — individual request within a pack
  - `messages` must be non-empty
  - `session_id` groups requests for sequential execution
- **`BenchmarkArtifact`** — the standard benchmark output container
  - `artifact_version == "2"` is a persisted schema contract
- **`BenchmarkSummary`** — aggregated metrics from a run
- **`BenchmarkRequestResult`** — per-request outcome with timing

**Hard rule**: Add optional fields with defaults, never remove or rename existing fields.

### Dataset loading: the better workload path

`benchmarks/datasets.py` is more valuable than template workloads. Key decisions:

1. **Reservoir sampling**: Deterministic with seeded RNG. Sort by original index after sampling for stable ordering. This matters for reproducibility.

2. **ShareGPT adapter**: Trims final assistant turn and uses its length to estimate `max_tokens`. This produces realistic output length distributions instead of fixed values.

3. **HuggingFace adapter**: Three-stage fallback — OpenAI messages format → ShareGPT conversation format → prompt/response field detection. Add new field names to `_PROMPT_FIELDS` / `_RESPONSE_FIELDS` when datasets use non-standard names.

4. **JSONL loader**: Preserves all WorkloadRequest fields including tools, headers, extra_body. This is the path for custom production traces.

### What to add for SkyPilot collaboration

#### 1. Auto-detect GPU type (P0)

Currently `--gpu` is manual. Add automatic detection:

```python
# Location: src/inferscope/hardware/detector.py (already exists but may need GPU detection)
# Approach: parse nvidia-smi output or use pynvml
# Fallback: if detection fails, log warning and continue without GPU metadata
# Never make GPU detection a blocking requirement — benchmarks work without it
```

#### 2. Cloud artifact upload (P1)

Add `--output-uri` flag to `inferscope benchmark`:
- `s3://bucket/path/` → upload via boto3
- `gs://bucket/path/` → upload via google-cloud-storage
- Local path → current behavior
- Make cloud SDKs optional dependencies (don't break core install)

```toml
# pyproject.toml
[project.optional-dependencies]
cloud = ["boto3>=1.34", "google-cloud-storage>=2.14"]
```

#### 3. Environment metadata in artifacts (P1)

Add to `BenchmarkArtifact`:
```python
environment: dict[str, Any] | None = None
# Should capture:
# - vllm_version or sglang_version (from /v1/models or pip)
# - cuda_version
# - gpu_driver_version
# - python_version
# - inferscope_version
# - skypilot_task_id (if run via SkyPilot, from env var)
```

#### 4. Benchmark comparison across GPU generations (P2)

`compare_benchmark_artifacts()` already works for two artifacts. Add:
- Multi-artifact comparison (N artifacts, same workload, different GPUs)
- Normalize by GPU cost (if cost metadata is available)
- Output: cost-efficiency table (goodput per dollar per GPU-hour)

## Prometheus Telemetry: What to Fix

### Priority fix: continuous sampling during benchmark

Current state: one scrape before benchmark, one scrape after. This misses everything interesting.

Implementation plan:
1. Add `--metrics-interval-ms` flag (default 2000ms)
2. Launch a background async task that scrapes all metrics targets at the configured interval
3. Store as `metrics_timeseries: list[MetricSnapshot]` in the artifact
4. Cancel the background task when the benchmark completes
5. Location: `benchmarks/prometheus_capture.py` (add `continuous_capture` coroutine)

Key metrics to track in time-series:
- `vllm:gpu_cache_usage_perc` — KV cache pressure over time
- `vllm:num_requests_running` + `waiting` — queue depth dynamics
- `vllm:num_preemptions_total` — preemption spikes during load ramp
- `vllm:gpu_prefix_cache_hit_rate` — cache warming behavior
- `sglang:token_usage` — SGLang equivalent

### Telemetry architecture

From `telemetry/prometheus.py`:
- Raw scrape preserves labels in `ScrapeResult.samples` as `list[MetricSample]`
- `raw_metrics` remains convenience-flattened by metric name (for backward compat)
- Engine detection is prefix-based (`vllm`, `sglang`, `atom`, `dynamo`)
- Metrics URL resolution supports either base endpoint or explicit `/metrics`
- Use label-preserving `samples_for()` / `samples_with_prefix()` accessors where per-worker granularity matters

## Workload Honesty: Implementation Rules

### hydration:template workloads

These YAMLs have a `hydration:template` tag and `target_context_tokens` metadata that vastly exceeds actual payload size. Rules:

1. **Never use template workloads as default** — prefer `--dataset sharegpt` as the default path
2. **Always emit a warning** when running a template workload (already done in `_build_runtime_warnings`)
3. **Never report `target_context_tokens`** as actual benchmark token counts in summaries
4. **Use `actual_context_tokens`** (computed from real message payload) for all runtime metrics

### When someone asks to "add a 128K context benchmark"

Do NOT create another template YAML with `target_context_tokens: 131072` and a 50-token stub message. Instead:

1. Point to `--dataset` mode with a real long-context dataset
2. Or point to `--context-file` for procedural expansion with real corpus data
3. Or use `--synthetic-input-tokens 131072` which at least pads to the target size (even if the padding is repeated blocks)

## Launcher/Planning Boundaries

From `benchmarks/launchers.py`:

- **`BenchmarkStackPlan`** = declarative planned stack (components, configs, env)
- **`MaterializedBenchmarkStack`** = written scripts/configs/env files on disk
- Materialization does NOT guarantee vendor binary/flag correctness on real machines

Real behaviors:
- Readiness probes (HTTP/TCP) are generated for all component types
- Scripts, env files, and config files are written to disk
- Proxy placeholder commands (`<path-to-...>`) are rejected during materialization
- Dependency ordering is enforced via `depends_on` chains

Maturity levels:
- **vLLM/SGLang single-endpoint**: most reliable benchmark path
- **vLLM disaggregated prefill/decode**: launch scripts generated, not validated on hardware
- **Dynamo**: planning-only, requires code-and-hardware validation before public claims

## Security and Endpoint Rules

From `security.py`:

- `validate_endpoint()` blocks private/localhost IPs by default (`allow_private=False`)
- `allow_private=True` is an explicit local-operator escape hatch
- CLI paths default to `allow_private=True` (local development convenience)
- MCP paths default to `allow_private=False` (network-exposed tool safety)
- Endpoint URL is scheme-validated and private-range aware (RFC 1918 + loopback)
- Trailing `/` is stripped from validated endpoints

**Hard rule**: Never weaken the MCP default. Add `allow_private` as an opt-in parameter instead.

## Testing Conventions

### What every new feature needs

1. **Unit test**: pure function behavior, no I/O, no network
2. **Mock transport test**: HTTP behavior using `httpx.MockTransport`
3. **Artifact round-trip test**: verify serialization → deserialization preserves all fields

### Test infrastructure

- `conftest.py` adds `src/` to `sys.path` — no editable install required
- Async tests use `pytest.mark.asyncio`
- HTTP mocking uses `httpx.MockTransport`, not `unittest.mock.patch`
- Test files follow `test_<module>_<area>.py` naming

### Change checklist

- If runtime scheduling changes → update `test_benchmark_runtime.py`
- If workload metadata semantics change → update dataset/model tests
- If telemetry wording changes → verify existing telemetry tests still match
- If new engine adapter added → add detection + config tests

## Common Mistakes to Avoid

1. **Don't add `approx_context_tokens` to new workloads** — use `target_context_tokens` (the renamed, honest version) and always set `hydration:template` or `hydration:hydrated` tag

2. **Don't use `allow_private=False` for CLI paths** — CLI runs locally, private endpoints must work. Only MCP paths restrict to public endpoints.

3. **Don't add new engines without a real adapter** — Dynamo's empty adapter (`return False`, `return {}`) is the cautionary tale. If you can't test it against a real endpoint, don't claim support.

4. **Don't duplicate `_resolve_benchmark_plan`** — CLI and MCP already share this function. Keep it that way.

5. **Don't break the `WorkloadPack` / `BenchmarkArtifact` contracts** — these are the stable API. Add optional fields with defaults, never remove or rename existing fields.

6. **Don't put SkyPilot-specific code in the benchmark runtime** — SkyPilot task YAMLs call InferScope's CLI. The runtime doesn't know or care about SkyPilot.

## Dependency Direction (Enforced)

```
hardware ─┐
models ───┤
           ├──→ optimization ──→ engines
           │          │
           │          ▼
           ├──→ telemetry ──→ profiling
           │          │
           │          ▼
           └──→ benchmarks ──→ tools
                      │
                      ▼
               cli*.py / server*.py
```

Hard rules:
- `optimization/` NEVER imports from `benchmarks/`
- `telemetry/` NEVER imports from `benchmarks/` or `optimization/`
- `benchmarks/` CAN import from `telemetry/` and `optimization/`
- SkyPilot integration is EXTERNAL — no `skypilot` imports inside `src/inferscope/`

## Related Documents

- [Root CLAUDE.md](../../CLAUDE.md) — strategic context, partnerships, priority stack
- [AGENTS.md](../../AGENTS.md) — directory routing, dev setup, test conventions
- [AUDIT-BENCHMARK-REALITY.md](docs/AUDIT-BENCHMARK-REALITY.md) — detailed honesty audit
- [docs/STRATEGIC-INTELLIGENCE.md](../../docs/STRATEGIC-INTELLIGENCE.md) — infrastructure partners, GPU access, competitive landscape

## Update This File When

These source files change materially:
- `src/inferscope/benchmarks/runtime.py` — runtime invariants
- `src/inferscope/benchmarks/models.py` — type contracts
- `src/inferscope/benchmarks/datasets.py` — dataset loading paths
- `src/inferscope/benchmarks/launchers.py` — launcher maturity
- `src/inferscope/telemetry/prometheus.py` — telemetry architecture
- `src/inferscope/security.py` — endpoint validation rules
- Benchmark/telemetry test conventions change
