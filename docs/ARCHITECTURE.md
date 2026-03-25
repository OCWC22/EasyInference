# ISB-1 Technical Architecture

This document describes the internal architecture of the ISB-1 benchmark system: module dependencies, data flow, and the execution lifecycle.

---

## Table of Contents

- [Module Overview](#module-overview)
- [Module Dependencies](#module-dependencies)
- [Data Flow](#data-flow)
- [Execution Lifecycle](#execution-lifecycle)
- [Key Abstractions](#key-abstractions)
- [Configuration System](#configuration-system)
- [Telemetry Pipeline](#telemetry-pipeline)

---

## Module Overview

ISB-1 is organized into four primary modules and several supporting modules:

| Module | Path | Responsibility |
|--------|------|---------------|
| **Workloads** | `workloads/` | Generate deterministic request traces for each workload type |
| **Harness** | `harness/` | Manage server lifecycle, execute benchmarks, collect data |
| **Analysis** | `analysis/` | Compute metrics, run statistical tests, generate visualizations |
| **Quality** | `quality/` | Evaluate output correctness against reference data |
| **Configs** | `configs/` | Hierarchical YAML configuration for GPUs, models, workloads, sweeps |
| **Publication** | `publication/` | Jinja2 templates for rendering results into reports |
| **Scripts** | `scripts/` | One-time setup, dataset download, trace generation |

---

## Module Dependencies

```
                    ┌──────────┐
                    │  configs  │
                    └────┬─────┘
                         │ (loaded by)
          ┌──────────────┼──────────────┐
          v              v              v
   ┌───────────┐  ┌───────────┐  ┌───────────┐
   │ workloads │  │  harness   │  │  quality   │
   └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
         │              │              │
         │  (traces)    │  (raw data)  │  (quality scores)
         └──────┬───────┘              │
                v                      │
         ┌───────────┐                 │
         │  analysis  │ <──────────────┘
         └─────┬─────┘
               │ (metrics, stats, plots)
               v
        ┌─────────────┐
        │ publication  │
        └─────────────┘
```

### Dependency Details

**workloads** depends on:
- `numpy` for random number generation and distributions
- `configs/workloads/*.yaml` for workload parameters
- `workloads/schemas/*.json` for tool-call schemas (agent workload)

**harness** depends on:
- `workloads` for loading pre-generated request traces
- `configs/` for GPU, model, workload, and sweep definitions
- `vllm` for the inference server and benchmark_serving client
- `requests` for health-check polling
- `harness.config_validator.ConfigValidator` for pre-run validation

**analysis** depends on:
- Raw result data produced by the harness
- `numpy`, `scipy` for metric computation and statistical tests
- `matplotlib`, `seaborn`, `plotly` for visualization

**quality** depends on:
- `rouge-score` for ROUGE-L evaluation
- `lm-eval` and `human-eval` (optional, for HumanEval and MMLU-Pro)
- Reference output data in `quality/reference_outputs/`

**publication** depends on:
- `jinja2` for template rendering
- Aggregated metrics and figures from the analysis module

---

## Data Flow

### Phase 1: Trace Generation (Offline)

```
configs/workloads/*.yaml
        │
        v
workloads.chat.ChatWorkloadGenerator
workloads.agent.AgentTraceGenerator
workloads.arrivals.PoissonArrival / GammaArrival
        │
        v
    traces/*.jsonl   (deterministic, seed-controlled)
```

Each workload generator reads its configuration, uses a seeded random number generator (`numpy.random.default_rng`), and writes a JSONL file where each line is a `Request` object serialized as JSON. Traces include messages in OpenAI chat-completion format, expected output token counts, session IDs, and metadata.

### Phase 2: Benchmark Execution (Online)

```
configs/sweep/core.yaml
        │
        v
harness.config_validator.ConfigValidator
        │ (validates)
        v
harness.sweep.SweepOrchestrator
        │
        ├── for each cell in matrix:
        │       │
        │       v
        │   harness.lockfile.LockfileGenerator  ──> lockfiles/*.json
        │       │
        │       v
        │   harness.server.VLLMServer
        │       │ .start() -> wait for /health 200
        │       │
        │       v
        │   harness.warmup.WarmupValidator
        │       │ (sends warmup requests, validates steady-state)
        │       │
        │       v
        │   harness.runner.BenchmarkRunner
        │       │
        │       ├── harness.client.BenchmarkClient.run(rate)
        │       │       │ (calls vllm benchmark_serving.py)
        │       │       v
        │       │   results/raw/<run_id>.json
        │       │
        │       ├── harness.telemetry.TelemetryCollector
        │       │       │ (GPU power, thermal, memory)
        │       │       v
        │       │   telemetry data (in-memory or file)
        │       │
        │       └── harness.engine_metrics.EngineMetricsCollector
        │               │ (scrapes /metrics endpoint)
        │               v
        │           engine metrics data
        │
        │       v
        │   harness.manifest.RunManifest  ──> results/raw/<run_id>/manifest.json
        │       │
        │       v
        │   harness.server.VLLMServer.stop()
        │
        └── (next cell)
```

### Phase 3: Analysis (Offline)

```
results/raw/*.json
        │
        v
analysis.metrics.MetricComputer
        │ (computes CellMetrics for each cell)
        v
results/aggregated/*.json, *.csv
        │
        ├── analysis.statistical.paired_ttest()
        │       (Mode A vs Mode B comparisons)
        │
        ├── analysis.statistical.bootstrap_ci()
        │       (confidence intervals on metrics)
        │
        ├── analysis.statistical.coefficient_of_variation()
        │       (trial stability checks)
        │
        ├── analysis.claim_evaluator
        │       v
        │   results/claims/*.json
        │
        ├── analysis.leaderboard
        │       v
        │   results/leaderboard/*.json
        │
        └── analysis.plots.*
                v
            publication/figures/*.png, *.html
```

### Phase 4: Publication (Offline)

```
results/aggregated/     ─┐
results/claims/         ─┤
results/leaderboard/    ─┤
publication/figures/    ─┤
                         v
            jinja2 template rendering
                         │
                         v
            publication/output/
                ├── whitepaper.md
                ├── blog_post.md
                └── claim_report.md
```

---

## Execution Lifecycle

### Single Cell Execution

The `BenchmarkRunner` manages the lifecycle of a single cell execution:

1. **Configuration resolution.** Load GPU, model, and workload configs. Resolve tensor parallelism, quantization, and memory settings.

2. **Lockfile generation.** `LockfileGenerator` captures vLLM version, CUDA version, PyTorch version, nvidia-smi output, NVLink topology, pip freeze, config file hashes, model revision, and random seeds.

3. **Server startup.** `VLLMServer.start()` spawns a `vllm serve` process, then polls the `/health` endpoint every 5 seconds until it returns HTTP 200, or until the 600-second startup timeout expires.

4. **Warmup.** `WarmupValidator` sends warmup requests and monitors throughput variance across sliding windows. Steady-state is declared when CV drops below 20%. Up to 3 extensions are allowed.

5. **Measurement.** For each rate in the workload's rate sweep:
   - `BenchmarkClient.run()` invokes `vllm.benchmarks.benchmark_serving` with the configured parameters.
   - `TelemetryCollector` samples GPU power/thermal readings in a background thread.
   - `EngineMetricsCollector` scrapes the `/metrics` Prometheus endpoint.
   - Results are written as JSON to `results/raw/`.

6. **Manifest.** `RunManifest` records run identity, timestamps, hardware, model, workload, mode, quantization, trial number, request counts, duration, and status.

7. **Shutdown.** `VLLMServer.stop()` sends SIGTERM to the process group, waits up to 30 seconds for graceful shutdown, then sends SIGKILL if necessary.

8. **Cooldown.** 30 seconds between consecutive rate points.

### Sweep Orchestration

The `SweepOrchestrator` iterates the full cross-product of the sweep matrix:

```
for gpu in gpus:
  for model in models:
    for workload in workloads:
      for mode in modes:
        for quantization in quantizations:
          for trial in range(num_trials):
            run_cell(gpu, model, workload, mode, quantization, trial)
```

After each trial completes, the orchestrator checks the CV threshold. If the CV of the primary metric across completed trials exceeds 10%, additional trials are scheduled up to the maximum.

---

## Key Abstractions

### Request (workloads.base)

An immutable dataclass representing a single inference request in OpenAI chat-completion format. Fields: `request_id`, `messages`, `expected_output_tokens`, `session_id`, `metadata`.

### WorkloadGenerator (workloads.base)

Abstract base class for all workload trace generators. Subclasses implement `generate(num_requests)` to produce a list of `Request` objects. The base class provides JSONL persistence via `save()` and `load()`.

### CellConfig / RunResult (harness.runner)

`CellConfig` encapsulates all parameters for a single benchmark cell. `RunResult` wraps the outcome including the result file path, manifest, and any errors.

### CellMetrics (analysis.metrics)

A dataclass holding all computed metrics for a single cell: latency percentiles (TTFT, TPOT, ITL, E2E), throughput, goodput, SLO attainment, engine metrics, power metrics, and error rates.

### MetricComputer (analysis.metrics)

Computes `CellMetrics` from raw per-request latency data, engine metric time-series, and GPU telemetry data. Enforces the invariant that TTFT is excluded from TPOT and ITL.

### RunManifest (harness.manifest)

An immutable metadata record for a single benchmark run. Serializable to/from JSON. Captures run identity, hardware, model, parameters, timing, and outcome status.

### LockfileGenerator (harness.lockfile)

Captures the complete software, hardware, and configuration state for reproducibility. Collects vLLM version, CUDA version, PyTorch version, nvidia-smi output, pip freeze, config file hashes, and model revision.

### ValidationResult (harness.config_validator)

Aggregated result of a configuration validation pass. Contains a list of errors and warnings with a boolean `ok` property.

---

## Configuration System

ISB-1 uses a hierarchical YAML configuration system:

```
configs/
├── gpus/          # Hardware definitions (one file per GPU)
├── models/        # Model specs with min GPU requirements and topologies
├── workloads/     # Workload parameters, arrival models, SLO gates
├── quality/       # Quality evaluation settings
└── sweep/         # Matrix definitions combining GPUs x models x workloads x modes
```

The `ConfigValidator` class loads, caches, and cross-validates configs. Key cross-validations:

- **Memory fit:** Estimates model VRAM at the requested quantization (using bytes-per-parameter lookup) and compares against available HBM with a 25% KV cache overhead factor.
- **Quantization support:** Verifies the GPU's `fp_formats` list includes the requested format.
- **Min GPU count:** Checks the model's `min_gpus` table against the cell's GPU count.

---

## Telemetry Pipeline

ISB-1 collects three categories of telemetry during benchmark execution:

### GPU Telemetry (harness.telemetry)

Collected via `nvidia-smi` or NVML:
- Power draw (watts)
- GPU temperature
- Memory utilization
- GPU utilization

Sampled at regular intervals during measurement. Used to compute `avg_power_watts` and `watts_per_token`.

### Engine Metrics (harness.engine_metrics)

Scraped from vLLM's Prometheus `/metrics` endpoint:
- `kv_cache_utilization` -- fraction of KV cache memory in use
- `prefix_cache_hit_rate` -- fraction of prefix cache hits
- `preemptions` -- cumulative preemption count
- `queue_depth` -- number of requests waiting to be scheduled

### Client Metrics (harness.client)

Collected by `benchmark_serving.py` per request:
- `ttft` -- time to first token
- `e2e_latency` -- end-to-end latency
- `output_tokens` -- number of output tokens
- `token_timestamps` -- per-token arrival timestamps (when available)
- `error` -- whether the request failed

All telemetry streams are time-aligned and passed to `MetricComputer` for aggregation.
