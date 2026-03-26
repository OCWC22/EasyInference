# Benchmark Subsystem Audit: What's Real, What's Not

**Date**: March 26, 2026  
**Scope**: Full audit of InferScope benchmark subsystem — workloads, experiment specs, launchers, runtime, telemetry, strategy, and MCP/CLI surfaces.

---

## Executive Summary

InferScope has a **real, functional HTTP benchmark runner** and **real Prometheus capture** wrapped in a much larger planning/catalog/deployment story that is **materially overstated**.

| Layer | Verdict | Confidence |
|-------|---------|------------|
| Benchmark runtime (HTTP replay, TTFT, TPOT, artifacts) | **Real — production-adjacent** | High |
| Prometheus scrape + normalize | **Real but shallow** | High |
| Multi-target metrics capture | **Real** | High |
| CLI/MCP surfaces | **Real — end-to-end callable** | High |
| Workload payloads | **Synthetic stubs, not real long-context data** | Critical gap |
| Experiment specs | **Planning metadata, not deployment proof** | Medium |
| Dynamo launcher | **Command sketch, not operational** | Low |
| LMCache launcher | **Partial — generates real configs, proxy is placeholder** | Medium |
| Dynamo adapter | **Stubbed — detect/metrics/config all return empty** | Not real |
| Engine compilers | **Flag assemblers with heuristic notes** | Medium |
| Strategy planner | **Useful routing logic, advisory not executable** | Medium |

**Bottom line**: ~40% real execution system, ~35% useful planning metadata, ~25% aspirational positioning.

---

## 1. Workloads: Synthetic Stubs, Not Real Long-Context Data

### Evidence

**`long-context-kv-offload-rag.yaml`** (lines 28–33):
```yaml
messages:
  - role: user
    content: |
      <retrieved_corpus>
      Imagine this block contains 96,000 tokens of incident tickets...
      </retrieved_corpus>
```

The actual message sent to the model is ~50 tokens. The `approx_context_tokens: 96000` metadata is used only for:
- SLO threshold bucketing (`runtime.py` line 324)
- Token count reporting (`runtime.py` line 136–138)
- Memory planning estimates (`support.py` line 86–88)

**It does NOT inflate the actual HTTP payload.** The model sees the literal YAML text.

**`coding-long-context.yaml`** (lines 26–34):
```yaml
- role: system
  content: |
    # === CODEBASE START ===
    src/server.py: benchmark entrypoints and replay orchestration
    src/cache.py: KV cache planner and allocator
    src/telemetry.py: latency and power collection
    # === CODEBASE END ===
```

Claims `approx_context_tokens: 32768`. Actual content: ~40 tokens.

### Procedural expansion is better but still synthetic

`procedural.py` can expand `tool-agent` and `coding-long-context` workloads:
- Repeats canned code blocks (`_CODING_CONTEXT_BLOCKS`, ~30 lines each)
- Shuffles from 5 canned prompts (`_TOOL_AGENT_PROMPTS`)
- Can read a `--context-file` for CLI (real data injection point)
- Expands by repeating blocks to hit target token count

This is **synthetic padding**, not realistic inference traffic. Real long-context benchmarks need:
- Actual document corpora (legal briefs, codebases, research papers)
- Realistic retrieval-augmented evidence sets
- Real multi-turn conversation histories
- Token counts that match the actual payload, not metadata claims

### What we actually have vs what we claim

| Workload | Claimed Tokens | Actual Payload Tokens | Gap |
|----------|---------------|----------------------|-----|
| `long-context-kv-offload-rag` | 96,000 | ~150 | 640× |
| `coding-long-context` | 32,768 | ~80 | 410× |
| `tool-agent` | N/A | ~200 | Honest |
| `enterprise-coding-agent` | N/A | ~500 | Honest |

---

## 2. Runtime: This Is Real

`benchmarks/runtime.py` (919 lines) is a genuine HTTP benchmark runner:

- **Sends real HTTP requests** via `httpx` with streaming SSE parsing
- **Supports 3 backends**: OpenAI chat, OpenAI completions, TRT-LLM generate_stream
- **Measures real timing**: TTFT from first SSE chunk, per-chunk ITL gaps, wall time
- **Session grouping**: requests in the same session run sequentially, different sessions run concurrently
- **Arrival scheduling**: Poisson, Gamma, or immediate arrivals with deterministic seeding
- **Warmup requests**: configurable warmup before measurement
- **Goodput/SLO**: per-request SLO thresholds with tiered bucketing by prompt length
- **Tool-call validation**: regex-based tool call detection for MCP workloads
- **Real artifacts**: `BenchmarkArtifact` with full request-level results, summary stats, metrics snapshots

### Measurement limitations (honest)

- TTFT = first SSE chunk arrival, not guaranteed first token
- ITL = inter-chunk gaps, not inter-token gaps
- TPOT = `(elapsed - ttft) / (completion_tokens - 1)` — approximate
- Runtime warns about these: `"ITL is approximated from streamed output chunks rather than token-level traces."`
- Only before/after Prometheus snapshots, no continuous time-series during the run

**Verdict**: Real and usable. Not lab-grade precision, but honest operator-level measurements.

---

## 3. Telemetry: Real but Lossy

### What works

`telemetry/prometheus.py` genuinely:
- Scrapes `/metrics` endpoints via HTTP
- Parses Prometheus text exposition format
- Detects engine from metric prefixes (vLLM, SGLang, ATOM, Dynamo)
- Normalizes into shared schema

`telemetry/capture.py` genuinely:
- Captures multiple metrics targets (primary, router, prefill, decode)
- Attaches before/after snapshots to benchmark artifacts
- Validates expected engine vs detected engine

### Critical flaw: label dimension flattening

`prometheus.py` line 258:
```python
result.raw_metrics[sample.name] = sample.value
```

**Same metric name with different labels overwrites.** For Dynamo per-worker metrics like:
```
dynamo_frontend_worker_queue_depth{worker="prefill-0"} 3
dynamo_frontend_worker_queue_depth{worker="decode-0"} 7
```
Only the last value (7) survives in `raw_metrics`.

### Label parsing is fragile

`parse_prometheus_text` splits labels on commas:
```python
for pair in labels_str.split(","):
```

This breaks on labels containing commas in values (e.g., `{path="/v1/chat/completions,stream=true"}`).

### Only snapshots, not time-series

Metrics are captured once before and once after the benchmark run. This misses:
- Peak queue depth during load
- Transient KV cache pressure
- Burst preemption events
- Time-correlated latency/throughput behavior

---

## 4. Dynamo Integration: Planning Layer, Not Operational

### DynamoAdapter is stubbed (`engines/dynamo.py` lines 62–73)

```python
class DynamoAdapter(EngineAdapter):
    async def detect_engine(self, endpoint: str) -> bool:
        return False  # Phase 5
    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        return {}
    async def get_config(self, endpoint: str, ...) -> dict[str, Any]:
        return {}
```

Every method returns empty. Engine detection, metrics collection, and config introspection are all placeholders.

### DynamoCompiler generates skeleton config

The generated Dynamo config (`_dynamo_config_files`) is extremely minimal:
```yaml
slo_planner:
  enabled: true
kv_router:
  enabled: true
nixl:
  enabled: true
grove:
  enabled: true
```

A real Dynamo deployment config requires worker endpoints, model paths, TP/PP settings, NIXL transport configuration, health check intervals, etc.

### Launcher generates plausible but unvalidated commands

The Dynamo launcher branch (`launchers.py` line 1114+) generates:
1. `dynamo serve --config configs/dynamo-config.yaml` — Smart Router
2. `vllm serve <model> --host ... --port 7100` — prefill worker
3. `vllm serve <model> --host ... --port 7200` — decode worker

These are reasonable command shapes, but:
- Worker registration with the router is not wired
- NIXL transport configuration between workers is not specified
- Health checks / readiness gates are absent
- No validation that `dynamo` binary exists or flags are correct

### What actually works for Dynamo today

1. ✅ Strategy correctly selects Dynamo lanes for NVIDIA multi-GPU
2. ✅ Experiment specs define correct multi-target metrics topology
3. ✅ Support gating correctly tiers Dynamo (supported/recommended)
4. ✅ Telemetry can detect Dynamo metrics prefixes
5. ✅ Normalizer groups Dynamo metrics by prefix family
6. ❌ Cannot actually launch a Dynamo deployment
7. ❌ Cannot detect a running Dynamo instance
8. ❌ Cannot introspect Dynamo config from a live endpoint

---

## 5. LMCache Integration: Partially Real

### What's real

The LMCache launcher (`launchers.py` lines 160–190, 751–917) generates:
- Real LMCache config YAMLs for prefiller and decoder nodes
- `kv_connector: LMCacheConnectorV1` with `kv_role: kv_producer/kv_consumer`
- `LMCACHE_CONFIG_FILE` env var wiring
- UCX transport configuration (`UCX_TLS=cuda_ipc,cuda_copy,tcp`)

### What's not real

The vLLM disaggregated proxy (`launchers.py` lines 878–906):
```python
"python3 <path-to-vllm-disagg-proxy> "
```

**Literal placeholder path.** The warning says:
> "Bundle is not runnable until a concrete vLLM disaggregation proxy command is provided."

So LMCache config generation is real, but the full stack isn't runnable without operator intervention.

---

## 6. Strategy Planner: Good Routing Logic, Advisory Output

`strategy.py` (628 lines) does real work:
- Selects workload mode from model/GPU/workload signals
- Routes to correct experiment lanes based on platform traits
- Correctly prioritizes Dynamo for NVIDIA, SGLang for agent workloads
- Handles Grace/non-Grace, RDMA/non-RDMA branching
- Multi-node vs multi-GPU distinction is correct

But the output is advisory:
- Returns lane descriptions, not executable deployment plans
- "Required" flag on lanes doesn't mean "runnable"
- No validation that selected experiments can actually be launched

---

## 7. Coverage Matrix: What We Can Actually Benchmark Today

### Fully functional paths (can run today against a live endpoint)

| Path | Workload | Engine | What It Measures |
|------|----------|--------|-----------------|
| Single-endpoint vLLM | Any packaged workload | vLLM | TTFT, TPOT, ITL, throughput, goodput |
| Single-endpoint SGLang | Any packaged workload | SGLang | Same |
| Single-endpoint TRT-LLM | Any packaged workload | TRT-LLM | Same (via generate_stream) |
| Procedural tool-agent | Synthetic MCP sessions | Any | Tool-call parse rate + above |
| Procedural coding | Synthetic code review | Any | Session reuse + above |

### Partially functional (requires operator setup)

| Path | What's Missing |
|------|---------------|
| vLLM disagg with LMCache | Proxy command placeholder — operator must supply path |
| vLLM disagg with NIXL | Same proxy issue |
| Dynamo disagg | Entire deployment — config, worker registration, binary |

### Not functional

| Path | Status |
|------|--------|
| Dynamo auto-detection | Stubbed (`return False`) |
| Dynamo config introspection | Stubbed (`return {}`) |
| Real 96K RAG workload | Payload is ~150 tokens |
| Real 32K coding workload | Payload is ~80 tokens |
| Continuous metrics during benchmark | Only before/after snapshots |

---

## 8. Hopper vs Blackwell: What's Actually Different

### ISA gating is real

`gpu_profiles.py` correctly maps:
- H100 SXM → `sm_90a`, H100 PCIe → `sm_90`
- B200/GB200 → `sm_100`, B300/GB300 → `sm_103`

Support gating uses these ISAs for experiment eligibility.

### But benchmarks don't exercise ISA differences

Nothing in the benchmark runner, workloads, or experiment specs actually:
- Uses different quantization formats per ISA (MXFP4/MXFP6 on Blackwell vs FP8 on Hopper)
- Configures different chunked prefill sizes per architecture
- Tests NVFP4 KV cache on Blackwell vs FP8 on Hopper
- Exercises Blackwell's 2× decode throughput advantage

The ISA data is used for **gating and labeling**, not for **architecture-aware benchmark configuration**.

---

## 9. Recommendations

### Immediate (make the existing system honest)

1. **Rename workload token claims** — Change `approx_context_tokens` to `target_context_tokens` and add `actual_payload_tokens` that reflects the real message size. Or better: add a `hydration_required: true` flag so the system doesn't pretend synthetic stubs are real benchmarks.

2. **Add real corpus hydration** — The `--context-file` path in `procedural.py` is the right idea. Extend it to all long-context workloads. Ship a downloader for standard corpora (PG19, RedPajama samples, etc.) or accept `--corpus-dir`.

3. **Fix Prometheus label flattening** — Change `raw_metrics` to `dict[str, list[tuple[dict[str, str], float]]]` or similar to preserve label dimensions. This is essential for Dynamo per-worker metrics.

4. **Mark Dynamo launcher as experimental** — Add a clear `experimental: true` flag to Dynamo stack plans. Don't let users think generated bundles are runnable without significant operator work.

5. **Fix vLLM disagg proxy placeholder** — Either integrate with vLLM's actual `vllm.entrypoints.openai.api_server` disaggregated mode, or document the exact script path.

### Short-term (make Dynamo actually work)

6. **Implement DynamoAdapter** — `detect_engine` should check for Dynamo metric prefixes. `get_config` should query Dynamo's config API if one exists.

7. **Wire Dynamo worker registration** — The launcher should generate complete Dynamo configs with worker endpoints, not just `enabled: true` flags.

8. **Add continuous metrics sampling** — Sample `/metrics` at configurable intervals during the benchmark run, not just before/after. This is critical for understanding KV cache pressure dynamics.

### Medium-term (make benchmarks real)

9. **Ship real workload corpora** — Partner with established benchmark datasets. For coding: use real repository snapshots. For RAG: use real document collections. The benchmark is only as good as its workload.

10. **Architecture-aware experiment configs** — Different chunked prefill sizes, KV quantization formats, and batch sizes for Hopper vs Blackwell. The ISA data exists; use it to drive configuration, not just labeling.

---

## 10. What We Can Honestly Say Today

### True statements

- "InferScope can benchmark any OpenAI-compatible inference endpoint with real HTTP traffic, streaming TTFT/TPOT/ITL measurement, and Prometheus metrics capture"
- "InferScope plans benchmark strategies across vLLM, SGLang, Dynamo, and TRT-LLM with GPU/ISA-aware support gating"
- "InferScope generates deployment stack plans with engine-specific launch commands and config files"
- "InferScope supports multi-target metrics capture for disaggregated topologies"

### Overstated claims to fix

- ~~"Realistic long-context RAG benchmarks"~~ → "Synthetic long-context scenario templates with configurable corpus injection"
- ~~"Dynamo disaggregated serving benchmark"~~ → "Dynamo experiment planning and metrics capture (deployment not automated)"
- ~~"Out-of-the-box 96K token RAG workload"~~ → "RAG scenario template — requires real corpus via `--context-file`"
- ~~"KV cache pressure simulation"~~ → "KV cache metrics capture from live endpoints (no synthetic pressure generation)"

---

## 11. Remediation Plan

### Priority order

| # | Task | Effort | Value | Dependencies | Status |
|---|------|--------|-------|-------------|--------|
| 1 | Fix Prometheus label flattening | 2–4 days | Critical | None | ✅ Done — quote-aware parser + label-preserving normalization |
| 2 | Honest workload metadata (template vs hydrated) | 1–2 weeks | High | None | ✅ Done — `HydrationMode`, `target_context_tokens`/`actual_context_tokens` split |
| 3 | Pin upstream vLLM+LMCache version, capture golden config | 1 week | Blocking | Hardware access | ⏳ Pending hardware |
| 4 | Make vLLM+LMCache disagg launcher runnable | 2–4 weeks | High | #3 | ⚠️ Partial — readiness probes added, proxy entrypoint defaulted |
| 5 | Implement DynamoAdapter.detect_engine() | 2–3 days | Medium | None | ✅ Done — uses shared Prometheus scrape |
| 6 | Pin Dynamo release, capture golden config | 1–2 weeks | Blocking | Hardware access | ⏳ Pending hardware |
| 7 | Make Dynamo launcher runnable (single-host static) | 4–8 weeks | High | #5, #6 | ⚠️ Partial — config gen, readiness probes, component ordering done |
| 8 | Update docs to separate template/validated/planning lanes | 1 week | High | After #2 | ✅ Done — docs updated with lane honesty |
| 9 | Add continuous metrics sampling during benchmark runs | 2–3 weeks | Medium | After #1 | ⏳ Pending |
| 10 | Real workload corpus hydration (ISB-1 integration) | 2–4 weeks | Medium | After #2 | ⏳ Pending |

### What's achievable in the next sprint

**Telemetry fix** (#1) and **workload honesty** (#2) can land immediately — no hardware needed, high value.

**DynamoAdapter detect** (#5) can land independently in a few days.

**Docs cleanup** (#8) can start immediately as an honesty-first pass.

### What requires hardware access

Items #3, #4, #6, #7 all require validated deployment on real Hopper/Blackwell hardware. These cannot be coded from documentation alone — the launcher must generate commands that have been proven to work on real metal.

### What is multi-month scope

True multi-node Dynamo orchestration, autoscaling, heterogeneous worker pools, and generalized backend support. These should be explicitly deferred and not promised in current docs.

### Lane classification

| Lane | Current Status | Target Status | When |
|------|---------------|---------------|------|
| Single-endpoint vLLM | ✅ Runnable | ✅ Runnable | Now |
| Single-endpoint SGLang | ✅ Runnable | ✅ Runnable | Now |
| Single-endpoint TRT-LLM | ✅ Runnable | ✅ Runnable | Now |
| vLLM disagg + LMCache | ⚠️ Planning only | ✅ Runnable | Sprint +2–4 weeks |
| vLLM disagg + NIXL | ⚠️ Planning only | ⚠️ Planning only | Deferred |
| Dynamo disagg + NIXL | ⚠️ Planning only | ✅ Runnable (single-host) | Sprint +4–8 weeks |
| Dynamo multi-node | 🔴 Aspirational | ⚠️ Planning only | Multi-month |

---

## 12. Benchmark Coverage: What We Actually Cover

### By workload type

| Workload | Template Quality | Hydration Path | KV Cache Relevance |
|----------|-----------------|---------------|-------------------|
| `tool-agent` | Good (real tool schemas) | Procedural expansion | Session reuse, prefix cache |
| `coding-long-context` | Stub (~80 tokens) | `--context-file` or procedural | Prefix cache, chunked prefill |
| `long-context-kv-offload-rag` | Stub (~150 tokens) | Needs `--context-file` | KV spill, cold session, overflow |
| `enterprise-coding-agent` | Good (~500 tokens) | No expansion yet | Multi-turn, tool-call parsing |
| `disagg-long-prompt` | Stub | No expansion | Disaggregated prefill stress |

### By measurement

| Metric | How Measured | Quality |
|--------|-------------|---------|
| TTFT | First SSE chunk timestamp | Good (operator-grade) |
| TPOT | (elapsed - ttft) / (completion_tokens - 1) | Approximate |
| ITL | Inter-chunk gaps | Approximate (not per-token) |
| Request throughput | requests / wall_time | Accurate |
| Output throughput | completion_tokens / wall_time | Accurate |
| Goodput | SLO-passing requests / wall_time | Real if SLOs configured |
| KV cache utilization | Prometheus scrape (before/after) | Snapshot only |
| Prefix cache hit rate | Prometheus scrape (before/after) | Snapshot only |
| Queue depth | Prometheus scrape (before/after) | Snapshot only |
| Tool-call parse rate | Regex pattern matching | Heuristic |

### By engine × topology

| Engine | Single-endpoint | Disaggregated P/D | Router/Orchestrated |
|--------|----------------|-------------------|-------------------|
| vLLM | ✅ Benchmark works | ⚠️ Launch plan only | N/A |
| SGLang | ✅ Benchmark works | ⚠️ Launch plan only | ⚠️ Launch plan only |
| Dynamo | N/A | ⚠️ Launch plan only | ⚠️ Launch plan only |
| TRT-LLM | ✅ Benchmark works | N/A | N/A |
| ATOM | ✅ Benchmark works | N/A | N/A |
