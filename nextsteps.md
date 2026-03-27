# EasyInference: Next Steps

Last updated: 2026-03-26

This document covers everything that needs to be done to finish building and testing the product before expanding benchmarks or pursuing compute partnerships.

---

## Current State Summary

### What Works

- **InferScope recommendation DAG** -- fully functional, runs locally without GPU. 14 models, 18 GPU profiles, 5 engine compilers.
- **InferScope MCP server** -- 28 tools registered across 6 groups (hardware intel, model intel, recommendations, KV cache, profiling, benchmarks). FastMCP wired correctly.
- **InferScope CLI** -- 20+ commands, all imports resolve. `recommend`, `validate`, `gpu`, `profile`, `benchmark-*` commands all present.
- **Engine compilers** -- vLLM (402 lines, Hopper/Blackwell/AMD-aware), SGLang (182 lines, RadixAttention/HiCache), ATOM (180 lines, AMD MLA/MoE). All production-quality.
- **Benchmark runtime** -- real async HTTP replay against OpenAI-compatible endpoints. Not a stub.
- **OpenAI replay wrapper** -- complete auth + replay integration layer.
- **ISB-1 workload generators** -- all 4 families (chat, agent, rag, coding) fully implemented with realistic content generation.
- **ISB-1 harness** -- runner, replay client, server management, sweep orchestrator, warmup validator, telemetry collector, engine metrics scraper, manifest/lockfile generation. All implemented.
- **ISB-1 analysis** -- MetricComputer (all metrics), statistical tests (paired t-test, BCa bootstrap), ResultAggregator, ClaimEvaluator (13 claims). All implemented.
- **ISB-1 quality modules** -- HumanEval, RULER, ROUGE, MMLU-Pro. All implemented but not wired into the benchmark pipeline.
- **ISB-1 visualization** -- 5 plot modules (throughput-latency, concurrency sweep, GPU telemetry, quality degradation, leaderboard heatmap). All implemented.
- **CI/CD** -- InferScope: lint, typecheck, security, package-smoke, test (Python 3.11/3.12/3.13). ISB-1: config validation, lint, format check, test (Python 3.10/3.12). Both should pass.
- **Security** -- SSRF protection, input validation, endpoint sanitization. All implemented.

### What's Broken or Missing

- **ITL metrics always zero** -- `replay_client.py` never populates `token_timestamps` during SSE parsing.
- **`LeaderboardGenerator`** -- empty stub (`class LeaderboardGenerator: pass`).
- **`ComparisonGenerator`** -- empty stub (`class ComparisonGenerator: pass`).
- **TRT-LLM adapter** -- stub (`detect_engine` returns `False`, `get_metrics` returns `{}`).
- **Dynamo adapter** -- stub (`detect_engine` returns `False`, `get_metrics` returns `{}`).
- **ISB-1 Mode B and Mode C configs** -- directories contain only `.gitkeep`.
- **ISB-1 quality reference outputs** -- directory contains only `.gitkeep`.
- **No `--external-endpoint` flag** -- ISB-1 harness can only benchmark a server it launches itself.
- **No live integration test file** -- `tests/integration/test_live_engines.py` referenced in CI but does not exist.
- **Silent error swallowing** -- `except Exception: pass` in sweep.py, replay_client.py, and engine adapters.
- **Dead API parameters** -- `target_latency_ms` and `target_ttft_ms` exposed but unused.
- **No economic metrics** -- no tok/$, tok/W in CellMetrics despite power data being collected.
- **Quality modules disconnected** -- exist but not called by the benchmark pipeline.
- **No real dataset support** -- ISB-1 uses only synthetic generation. No BurstGPT, ShareGPT, WildChat, or Mooncake trace integration.
- **No disaggregated serving benchmarking** -- ISB-1 harness only supports colocated single-server.
- **No MI350X GPU profile** -- shipped Q1 2026, missing from gpu_profiles.py.

---

## Phase 0: Critical Fixes (Week 1)

These must be fixed before anything else. They are bugs or empty stubs in core functionality.

### 0.1 Fix ITL timestamp collection

**File:** `products/isb1/harness/replay_client.py`
**Problem:** `token_timestamps` is initialized as an empty list and never populated during SSE stream parsing in `_run_stream_request`. Only `first_output_timestamp` is tracked.
**Impact:** ITL (Inter-Token Latency) metrics in `analysis/metrics.py` always compute as zero. ITL is the third core metric in the methodology document. Publishing results with `itl_p50: 0.0` destroys credibility.
**Fix:** Capture `time.perf_counter()` on each SSE `data:` chunk that contains output content. Append to `token_timestamps` on the `ReplayRequestResult`. Estimated: ~10 lines of code.

### 0.2 Implement LeaderboardGenerator

**File:** `products/isb1/analysis/leaderboard.py`
**Problem:** Contains only `class LeaderboardGenerator: pass`.
**Impact:** The `leaderboard` CLI command has no data source. The leaderboard heatmap plot is fully implemented but depends on this. Cannot rank benchmark configurations.
**Fix:** Implement class that loads aggregated results, sorts by configurable metric (default: generation_throughput), applies optional filters (GPU, model, workload), returns ranked table data. The `analyze` CLI command already produces the input data this needs.

### 0.3 Implement ComparisonGenerator

**File:** `products/isb1/analysis/comparisons.py`
**Problem:** Contains only `class ComparisonGenerator: pass`.
**Impact:** Cannot programmatically compare Mode A vs Mode B results. The entire "did optimization help?" workflow is broken. This is what goes into blog posts and sponsor reports.
**Fix:** Implement class that takes two sets of aggregated results, computes deltas and percentage changes for all metrics, runs paired t-tests for statistical significance, flags improvements and regressions. InferScope's `compare_benchmark_artifacts` in `benchmarks/catalog.py` already does this for InferScope artifacts -- use the same pattern.

### 0.4 Fix silent error swallowing

**Files:** `products/isb1/harness/sweep.py`, `products/isb1/harness/replay_client.py`, `products/inferscope/src/inferscope/engines/vllm.py`, `sglang.py`, `atom.py`
**Problem:** Multiple `except Exception: pass` blocks silently swallow YAML parse errors, streaming errors, and metrics scraping failures. On paid GPU compute, silent failures waste money and hide bugs.
**Fix:** Replace `pass` with `logger.warning(...)` at minimum. In `replay_client.py`, differentiate connection errors from parse errors from timeouts. In engine adapters, log which metric endpoint failed and why.

### 0.5 Kill dead API parameters

**Problem:** `target_latency_ms` and `target_ttft_ms` are exposed in tool/CLI interfaces but do nothing.
**Impact:** Users configure these expecting behavior. When nothing happens, trust erodes.
**Fix:** Either wire them into SLO evaluation in the recommendation DAG (connect to `ObjectiveSpec.ttft_p95_ms`) or remove them from the interface.

---

## Phase 1: Make It Testable Without GPUs (Week 1-2)

These changes enable testing the full pipeline on Modal, cloud endpoints, or any existing inference server.

### 1.1 Add --external-endpoint flag to ISB-1 runner

**File:** `products/isb1/harness/runner.py`
**Problem:** `BenchmarkRunner.run()` always launches its own `VLLMServer`. Cannot point at Modal, Fireworks, Baseten, or any existing deployment.
**Fix:** Add `external_endpoint: str | None = None` to `CellConfig`. If set, skip `VLLMServer` start/stop in `run()`, pass the URL directly to `BenchmarkClient`. Skip `TelemetryCollector` start (no local GPU). Keep `EngineMetricsCollector` if the external endpoint exposes `/metrics`. Estimated: ~30 lines in runner.py, ~5 lines in cli.py for the flag.

### 1.2 Add --external-endpoint to ISB-1 CLI

**File:** `products/isb1/harness/cli.py`
**Commands:** `run-cell` and `run` (sweep)
**Fix:** Add `--endpoint` option. When provided, set `CellConfig.external_endpoint` and skip server-related config validation (GPU topology checks still run for recommendation context).

### 1.3 Verify MCP server end-to-end

**Task:** Run `inferscope serve` locally, connect with a simple MCP client, call each tool group:
- `tool_get_gpu_specs("H100")` -- should return GPU profile
- `tool_recommend_config("DeepSeek-R1", "h100", "agent", 8)` -- should return full config
- `tool_list_benchmark_workloads()` -- should return 9 workload packs
- `tool_profile_runtime("http://localhost:8000")` -- should fail gracefully (no endpoint) with a useful error

**Fix any tools that crash or return malformed responses.**

### 1.4 Verify fresh-clone reproducibility

**Task:** Clone the repo into a fresh directory. Run:
```bash
cd products/inferscope && uv sync --dev && uv run pytest
cd products/isb1 && pip install -e ".[dev]" && python -m pytest tests/ -v
```
**Fix whatever breaks.** Previous validation flagged this as the "#1 docs credibility issue."

### 1.5 Fix mypy compliance

**File:** `products/inferscope/pyproject.toml` mypy config
**Task:** Run `uv run mypy src/inferscope/` and fix errors, OR relax the config to match what actually passes. Do not claim strict type safety that doesn't exist.

---

## Phase 2: Complete the Analysis Pipeline (Week 2-3)

These make benchmark results useful and publishable.

### 2.1 Add economic metrics to CellMetrics

**File:** `products/isb1/analysis/metrics.py`
**Add to `CellMetrics`:**
- `tokens_per_watt: float = 0.0` -- `generation_throughput / avg_power_watts`
- `tokens_per_dollar_hour: float = 0.0` -- `generation_throughput * 3600 / gpu_hourly_cost`
- `cost_per_million_tokens: float = 0.0` -- `(gpu_hourly_cost / 3600) / generation_throughput * 1e6`

**Add GPU cost table** (source: GPU-MARKET-MAP.md and public pricing):
```
H100 SXM: $2.00/hr (spot), $3.50/hr (on-demand)
H200 SXM: $3.50/hr (spot), $5.50/hr (on-demand)
B200: $4.00/hr (spot), $6.50/hr (on-demand)
MI300X: $1.50/hr (spot), $3.00/hr (on-demand)
MI355X: $2.50/hr (spot), $5.50/hr (on-demand)
```

Power data already collected via DCGM telemetry. This is purely a computation addition.

### 2.2 Wire quality modules into benchmark pipeline

**File:** `products/isb1/harness/runner.py`
**Problem:** HumanEval, RULER, ROUGE, MMLU-Pro exist as standalone modules but are never called by the benchmark runner. The `quality` CLI command is a placeholder.
**Fix:** After the replay phase in `run()`, optionally call quality modules against the served model endpoint. Store quality results in the run directory alongside performance results. Add a `--skip-quality` flag for speed when only performance is needed.

### 2.3 Populate ISB-1 Mode B configs

**Directory:** `products/isb1/configs/modes/mode_b/`
**Problem:** Contains only `.gitkeep`. Mode B is "optimized configuration" -- the InferScope recommendation output.
**Fix:** For each GPU/model combination in Mode A, run `inferscope recommend` and translate the output into Mode B config YAMLs. This directly demonstrates the value of InferScope: Mode A (default) vs Mode B (InferScope-optimized).

### 2.4 Generate quality reference outputs

**Directory:** `products/isb1/quality/reference_outputs/`
**Problem:** Empty. ROUGE evaluation compares against BF16 reference outputs that don't exist.
**Fix:** Generate BF16 reference outputs for each model in the benchmark matrix by running inference at BF16 precision and saving results. This requires a GPU endpoint (can be done on Modal).

---

## Phase 3: Real Dataset Integration (Week 3-4)

These replace synthetic-only generation with production-realistic workloads. This is the single most cited gap in existing benchmarking tools.

### 3.1 Add BurstGPT arrival pattern support

**Dataset:** BurstGPT (10.3M traces, 213 days, Azure OpenAI). Already integrated in vLLM.
**Source:** `github.com/HPMLL/BurstGPT`, HuggingFace `lzzmm/BurstGPT`
**Integration point:** `products/isb1/workloads/arrivals.py`
**Fix:** Add `BurstGPTArrival` class that loads real arrival timestamps from BurstGPT traces and replays them instead of synthetic Poisson/Gamma. Support filtering by model type (GPT-4 vs ChatGPT) and service type (API vs Conversation). This captures real burstiness (weekly periodicity, aperiodic bursts, Gamma-distributed concurrency) that synthetic models miss.

### 3.2 Add ShareGPT/WildChat as prompt sources

**Datasets:**
- ShareGPT V3 (53K conversations, universal benchmark standard, Apache-2.0)
- WildChat-4.8M (4.8M conversations with timestamps AND token counts, ODC-BY)

**Integration point:** `products/isb1/workloads/chat.py` (already has ShareGPT support), `materialize.py`
**Fix:** ISB-1's chat generator already accepts `sharegpt_path`. Extend to:
- Auto-download ShareGPT V3 if not present (like vLLM does)
- Add WildChat loader with timestamp-aware replay and per-turn token count metadata
- Add config option to select dataset source: `synthetic`, `sharegpt`, `wildchat`

### 3.3 Add Mooncake/Kimi traces for long-context workloads

**Dataset:** Mooncake/Kimi traces (FAST 2025 Best Paper). Production long-context chatbot, 100B+ tokens/day, 200K+ context.
**Source:** `github.com/kvcache-ai/Mooncake/tree/main/FAST25-release/traces`
**Integration point:** `products/isb1/workloads/rag.py`
**Fix:** Add `MooncakeTraceLoader` that imports JSONL traces with request-level timestamps, input/output token lengths, and prefix-sharing information. Use for long-context RAG benchmarks instead of synthetic bimodal generation. This is the most realistic long-context dataset available.

### 3.4 Add agentic dataset support

**Datasets:**
- BFCL (Berkeley Function Calling Leaderboard): 2,000 function-calling pairs
- SWE-bench trajectories (Nebius): 80K+ real multi-step agent workflows
- ToolBench: 16,000 APIs, 3 complexity levels

**Integration point:** `products/isb1/workloads/agent.py`
**Fix:** Add loaders for BFCL function-calling pairs (replaces synthetic tool schemas with real API definitions) and SWE-bench trajectories (replaces synthetic agent sessions with real coding agent workflows including growing context). Config option: `source: synthetic | bfcl | swebench | toolbench`.

### 3.5 Add coding dataset support

**Datasets:**
- CodeChat (82K coding conversations, 14:1 response:prompt ratio)
- DevGPT (29K prompts linked to GitHub commits)
- CrossCodeEval (cross-file context completion)

**Integration point:** `products/isb1/workloads/coding.py`
**Fix:** Add loaders for CodeChat (realistic coding conversation patterns with measured token ratios) and CrossCodeEval (realistic cross-file context that tests prefix caching). The current synthetic repo context is plausible but these are measured from real developer interactions.

---

## Phase 4: Disaggregated Serving Support (Week 4-5)

Almost every production LLM serving framework now runs disaggregated prefill/decode. A benchmark tool that only measures colocated serving is incomplete.

### 4.1 Add multi-server support to ISB-1 harness

**File:** `products/isb1/harness/server.py`
**Problem:** Only `VLLMServer` exists. Cannot launch SGLang, Dynamo, or disaggregated topologies.
**Fix:** Add `SGLangServer` class with same lifecycle interface. Add `DisaggregatedStack` class that manages a prefill server + decode server + optional router. The runner should accept a `server_type` config: `vllm`, `sglang`, `external`, `disaggregated`.

### 4.2 Add P/D split metrics to ISB-1

**File:** `products/isb1/analysis/metrics.py`
**Problem:** No metrics for disaggregated serving.
**Add to `CellMetrics`:**
- `prefill_node_utilization: float = 0.0`
- `decode_node_utilization: float = 0.0`
- `kv_transfer_latency_ms_p95: float = 0.0`
- `pd_handoff_latency_ms_p95: float = 0.0`

### 4.3 Implement Dynamo adapter

**File:** `products/inferscope/src/inferscope/engines/dynamo.py`
**Problem:** Adapter is a stub. `detect_engine` returns `False`, `get_metrics` returns `{}`. Dynamo 1.0 shipped March 16, 2026 and is deployed by AWS, Azure, Google Cloud, Baseten, Fireworks, Cursor, Perplexity.
**Fix:** Implement `detect_engine` (check for Dynamo-specific metrics or `/health` response), `get_metrics` (scrape Dynamo's Prometheus endpoint), `get_config` (query Dynamo's management API). The compiler already generates planning configs.

### 4.4 Add LMCache benchmark integration

**Problem:** InferScope has experiment specs for LMCache disaggregated lanes but nothing has been tested.
**Fix:** Add a `LMCacheStack` launcher to ISB-1 that can start a prefill endpoint + decode endpoint + LMCache connector. Support cache tiers: GPU -> CPU -> disk -> S3. Measure KV transfer latency and cache hit rates as first-class metrics.

---

## Phase 5: Hardware and Model Coverage (Week 5-6)

### 5.1 Add MI350X GPU profile

**File:** `products/inferscope/src/inferscope/hardware/gpu_profiles.py`
**Specs:** AMD MI350X, gfx950 (CDNA4), 288GB HBM3E, 8 TB/s bandwidth, native MXFP4, OCP FP8, ~5000 FP8 TFLOPS, 750W TDP. Available on Azure (ND-MI350X-v1), OCI, CoreWeave, Lambda. Priced at $5.50-$7.00/GPU-hour (25-30% cheaper than Blackwell).

### 5.2 Verify all model recommendations locally

**Task:** Run the following commands and verify output is sane (correct TP, precision, engine, memory fit):
```bash
inferscope recommend Kimi-K2.5 h100 --num-gpus 8 --workload agent
inferscope recommend Kimi-K2.5 b200 --num-gpus 8 --workload coding
inferscope recommend GLM-4.7 h100 --num-gpus 8 --workload long_context_rag
inferscope recommend GLM-4.7 b200 --num-gpus 4 --workload coding
inferscope recommend DeepSeek-R1 mi300x --num-gpus 8 --workload chat
inferscope recommend DeepSeek-R1 mi355x --num-gpus 8 --workload agent
inferscope recommend Qwen3.5-397B-A17B h200 --num-gpus 8 --workload chat
inferscope recommend Llama-3-70B h100 --num-gpus 2 --workload coding
```
**Fix any configs that produce invalid TP, don't fit in memory, or pick wrong engines.**

### 5.3 Expand model registry

**File:** `products/inferscope/src/inferscope/models/registry.py`
**Currently:** 14 models.
**Add:**
- Llama 4 Maverick (400B MoE, multimodal)
- Llama 4 Scout
- Gemma 3 (27B)
- Phi-4 (14B)
- Command R+ (104B)
- Qwen3-Coder (if released)

These are the models operators are deploying in Q1/Q2 2026.

### 5.4 Verify FlashAttention-4 references

**File:** `products/inferscope/src/inferscope/engines/vllm.py`
**Problem:** References FA4 features for Blackwell, but FA4 is currently forward-only with no GQA/MQA support and isn't in production vLLM yet.
**Fix:** Gate FA4 notes behind a version check or add a caveat. Do not present aspirational features as current capabilities.

---

## Phase 6: Testing (Week 5-7, parallel with Phase 5)

### 6.1 Add integration test smoke path for InferScope

**File:** `products/inferscope/tests/integration/test_live_engines.py` (referenced in CI, does not exist)
**Fix:** Create the file. Add tests that:
- Start a mock HTTP server returning vLLM-style `/metrics` and `/v1/models` responses
- Run `analyze_runtime()` against it
- Verify normalized metrics, health assessment, and workload classification
- Run `recommend()` and verify output is valid JSON

### 6.2 Add integration test for ISB-1 pipeline

**File:** `products/isb1/tests/test_runner_integration.py`
**Fix:** Create a test that:
- Starts a mock HTTP server that accepts `/v1/chat/completions` and returns streaming SSE
- Uses `--external-endpoint` (from Phase 1.1) to point the runner at it
- Runs the full pipeline: materialize -> replay -> aggregate -> metrics
- Verifies TTFT, TPOT, ITL, throughput, goodput are all non-zero and sane

### 6.3 Add tests for untested critical paths

**Priority order:**
1. `BenchmarkRunner.run()` with external endpoint (after 1.1)
2. `ResultAggregator` end-to-end
3. `SweepOrchestrator.execute()` with a single-cell sweep
4. At least one quality module (ROUGE is simplest)
5. At least one plot module (throughput-latency is simplest)

### 6.4 Add tests for ISB-1 token timestamp collection

**After fixing 0.1**, add a test in `test_replay_client.py` that:
- Sends a streaming response with 10 SSE chunks
- Verifies `token_timestamps` has 10 entries
- Verifies ITL gaps are computed correctly from those timestamps

---

## Phase 7: Open TCO and Competitive Differentiation (Week 7-8)

### 7.1 Build open TCO model

**Problem:** InferenceX uses SemiAnalysis's proprietary TCO model (surveying 70+ GPU clouds). This is the most cited competitive gap.
**Fix:** Build an open TCO model using public GPU pricing data:
- Spot and on-demand prices from Modal, Baseten, RunPod, Vast.ai, Lambda, CoreWeave, AWS, GCP, Azure
- Include power costs at standard datacenter rates
- Compute: cost per million tokens, cost per request at SLO, total cost of ownership per month at target QPS
- Publish methodology and data sources. "Open TCO" vs "proprietary TCO" is a clear differentiator.

### 7.2 Add Pareto frontier visualization

**File:** `products/isb1/analysis/plots/`
**Fix:** Add a plot that shows the throughput-latency-cost Pareto frontier across configurations. X-axis: generation throughput. Y-axis: TTFT P95. Color: cost per million tokens. Highlight Pareto-optimal configurations. This is becoming the standard way to present serving benchmark results.

### 7.3 Implement merge-trace approach

**Problem:** No single dataset has both realistic arrival patterns AND realistic prompt content.
**Fix:** Implement the standard industry approach: merge BurstGPT arrival patterns with ShareGPT/WildChat prompt distributions. Create a `MergedTraceGenerator` that:
1. Loads BurstGPT timestamps for arrival patterns
2. Loads ShareGPT/WildChat for prompt content
3. Matches requests by approximate token length
4. Produces a combined trace with real timing and real content

This is explicitly called "the standard industry approach" in serving papers and is the #1 cited gap in existing tools.

---

## Dependency Graph

```
Phase 0 (fixes)
  |
  v
Phase 1 (testable without GPU)
  |
  +--------+--------+
  v        v        v
Phase 2  Phase 3  Phase 6
(analysis) (datasets) (testing)
  |        |
  v        v
Phase 4 (disaggregated serving)
  |
  v
Phase 5 (hardware/model coverage)
  |
  v
Phase 7 (TCO and competitive differentiation)
```

Phases 2, 3, and 6 can run in parallel after Phase 1.
Phase 4 depends on Phase 2 (metrics) and Phase 3 (realistic workloads).
Phase 5 can start any time but full validation requires GPU access.
Phase 7 depends on Phases 2 and 3 for data.

---

## Time Estimate

| Phase | Duration | GPU Required? |
|-------|----------|---------------|
| Phase 0: Critical fixes | 3-5 days | No |
| Phase 1: Testable without GPU | 3-5 days | No |
| Phase 2: Analysis pipeline | 5-7 days | Partial (reference outputs need GPU) |
| Phase 3: Real datasets | 7-10 days | No |
| Phase 4: Disaggregated serving | 7-10 days | For validation only |
| Phase 5: Hardware/model coverage | 3-5 days | For validation only |
| Phase 6: Testing | 5-7 days | No (mock servers) |
| Phase 7: TCO and differentiation | 5-7 days | No |

**Total to demo-ready (Phases 0-1):** ~1-2 weeks, no GPU.
**Total to publish-ready (Phases 0-6):** ~6-8 weeks, GPU needed for Phases 2.4, 4, 5 validation.
**Total to competitive parity (all phases):** ~8-10 weeks.
