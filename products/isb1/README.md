# ISB-1: InferScope Benchmark Standard for LLM Serving Optimization

**Version 1.0.0** | **License: Apache 2.0**

ISB-1 is a rigorous, reproducible benchmark harness for evaluating large language model (LLM) inference serving performance. It measures throughput, latency, goodput, and quality across realistic workloads on modern GPU hardware, providing a standardized framework for comparing inference engines, optimization strategies, and hardware configurations.

Within the EasyInference monorepo, the benchmark product lives at `products/isb1/`.

---

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Benchmark Matrix](#benchmark-matrix)
- [CLI Reference](#cli-reference)
- [Configuration Reference](#configuration-reference)
- [Results Format](#results-format)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

ISB-1 addresses the need for a standardized, transparent, and reproducible benchmark for LLM inference serving. Unlike ad-hoc benchmarks that test a single model on a single GPU under synthetic conditions, ISB-1 defines:

- **Four production-representative workloads** spanning multi-turn chat, agentic tool calling, retrieval-augmented generation, and repository-scale coding assistance.
- **Three execution modes** that separate default engine behavior from vendor-optimized and operator-submitted configurations.
- **A complete measurement methodology** including warmup validation, steady-state detection, and statistical rigor through paired t-tests and bootstrap confidence intervals.
- **A quality evaluation track** ensuring that optimization does not degrade output correctness.
- **A claim evaluation framework** for validating vendor performance assertions against independently measured data.

ISB-1 targets the vLLM inference engine and uses its `benchmark_serving.py` harness as the underlying measurement tool. Results are collected via the OpenAI-compatible chat completions API, ensuring relevance to real-world deployment patterns.

### Key Design Principles

1. **Reproducibility first.** Every run produces a lockfile capturing the full software, hardware, and configuration state. Random seeds are fixed. Traces are deterministic.
2. **Statistical rigor.** Minimum 3 trials per cell. Coefficient of variation must be below 10%. High-variance cells are automatically flagged and extended to 5 trials.
3. **Production fidelity.** Workloads use realistic conversation patterns, context lengths, arrival distributions, and session structures drawn from production serving data.
4. **Separation of concerns.** The benchmark harness is decoupled from the inference engine, the analysis pipeline, and the publication layer.

---

## Quick Start

For a complete walkthrough, see [docs/QUICKSTART.md](docs/QUICKSTART.md).

### Prerequisites

- Python 3.10 or later
- NVIDIA GPU(s): H100 SXM, H200 SXM, B200, or B300
- CUDA 12.x with compatible driver
- vLLM installed and functional
- HuggingFace model access (gated models require `HF_TOKEN`)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd EasyInference/products/isb1

# Install the benchmark package
pip install -e ".[dev,quality]"

# Run node setup (installs telemetry, downloads datasets, generates traces)
make setup
```

### Run Your First Benchmark

```bash
# Validate configuration before running
isb1 validate --sweep configs/sweep/core.yaml

# View the execution plan
isb1 plan --config configs/sweep/core.yaml

# Run the core benchmark suite
isb1 run --config configs/sweep/core.yaml --output results/

# Or run a single cell for quick testing
isb1 run-cell \
  --gpu h100 \
  --model llama70b \
  --workload chat \
  --mode mode_a \
  --quantization fp8 \
  --output results/
```

### View Results

```bash
# Aggregate completed runs into a consolidated analysis file
isb1 analyze --results-dir results/ --output analysis.json

# Review publication-readiness / claim prerequisites
isb1 claims --results-dir results/

# Generate a leaderboard view
isb1 leaderboard --analysis analysis.json

# Render an HTML report
isb1 report --analysis analysis.json --output report.html
```

---

## Architecture Overview

ISB-1 is organized into four top-level modules, each with a clear responsibility boundary. For a detailed technical walkthrough, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

```
products/isb1/
├── workloads/         # Workload trace generation
│   ├── base.py        #   Abstract WorkloadGenerator, Request dataclass
│   ├── chat.py        #   WL-1: Multi-turn chat (ShareGPT + synthetic)
│   ├── agent.py       #   WL-2: Agentic tool-calling traces
│   ├── arrivals.py    #   Poisson and Gamma arrival-time generators
│   └── schemas/       #   Tool-call JSON schemas for agent workloads
├── harness/           # Benchmark execution engine
│   ├── server.py      #   VLLMServer lifecycle management
│   ├── client.py      #   BenchmarkClient (wraps benchmark_serving.py)
│   ├── runner.py      #   BenchmarkRunner: single-cell execution
│   ├── sweep.py       #   SweepOrchestrator: full matrix execution
│   ├── warmup.py      #   WarmupValidator: steady-state detection
│   ├── config_validator.py  # YAML validation and cross-checks
│   ├── manifest.py    #   RunManifest metadata record
│   ├── lockfile.py    #   LockfileGenerator: reproducibility snapshot
│   ├── telemetry.py   #   GPU power/thermal telemetry collection
│   └── engine_metrics.py    # vLLM engine metric scraping
├── analysis/          # Post-run analysis and statistics
│   ├── metrics.py     #   CellMetrics dataclass and MetricComputer
│   ├── statistical.py #   Paired t-test, bootstrap CI, CV checks
│   └── plots/         #   Visualization generators
├── quality/           # Output quality evaluation
│   ├── rouge_eval.py  #   ROUGE-L scoring against reference outputs
│   └── reference_outputs/   # Gold-standard reference data
├── configs/           # YAML configuration hierarchy
│   ├── gpus/          #   GPU hardware definitions
│   ├── models/        #   Model specifications and min GPU requirements
│   ├── workloads/     #   Workload parameters and SLO definitions
│   ├── quality/       #   Quality evaluation configurations
│   └── sweep/         #   Sweep matrix definitions (core, extended)
├── publication/       # Report generation
│   ├── templates/     #   Jinja2 templates for whitepaper, blog, claims
│   ├── figures/       #   Generated plots
│   └── tables/        #   Generated data tables
├── results/           # Output directory hierarchy
│   ├── raw/           #   Per-run JSON results
│   ├── aggregated/    #   Computed metrics per cell
│   ├── claims/        #   Claim evaluation reports
│   └── leaderboard/   #   Ranked leaderboard data
├── lockfiles/         # Reproducibility snapshots
├── scripts/           # Setup and utility scripts
└── tests/             # Test suite
```

### Data Flow

1. **Trace Generation.** Workload generators produce deterministic JSONL request traces with realistic conversation patterns, context distributions, and arrival times.
2. **Execution.** The harness starts a vLLM server with the cell's configuration, runs warmup validation, then executes the benchmark at each rate in the sweep. GPU telemetry and engine metrics are collected in parallel.
3. **Measurement.** Raw per-request latency data (TTFT, TPOT, ITL, E2E) is recorded alongside engine-level metrics (KV cache utilization, preemptions, queue depth) and GPU telemetry (power draw).
4. **Analysis.** The MetricComputer aggregates raw data into CellMetrics. Statistical tests compare modes and configurations. Claims are evaluated against measured deltas.
5. **Publication.** Jinja2 templates render the analyzed data into whitepaper, blog post, and claim report formats.

---

## Benchmark Matrix

### GPUs

| GPU | Short Name | HBM | FP Formats | NVLink |
|-----|-----------|-----|------------|--------|
| H100 SXM 80GB | `h100` | 80 GB | bf16, fp16, fp8 | Gen4, 900 GB/s |
| H200 SXM | `h200` | 141 GB | bf16, fp16, fp8 | Gen4, 900 GB/s |
| B200 | `b200` | 192 GB | bf16, fp16, fp8, nvfp4 | Gen5 |
| B300 | `b300` | 288 GB | bf16, fp16, fp8, nvfp4 | Gen5 |

### Models

| Model | Short Name | Architecture | Parameters | Min GPUs (fp8) |
|-------|-----------|-------------|------------|----------------|
| DeepSeek-R1 | `dsr1` | MLA + MoE | 671B | varies by GPU |
| Qwen2.5-235B | `qwen235b` | Hybrid MoE | 235B | varies by GPU |
| Llama-3.3-70B | `llama70b` | Dense GQA | 70B | 1 (H100+) |
| DeepSeek-V3.2 | `dsv32` | MLA + MoE | - | varies by GPU |
| Kimi-K2.5 | `kimik25` | MoE | - | varies by GPU |
| GLM-5 | `glm5` | Dense GQA | - | varies by GPU |
| Qwen3.5-35B | `qwen35b` | Dense GQA | 35B | varies by GPU |
| Llama-4-Maverick | `maverick` | MoE | - | varies by GPU |
| Qwen2.5-72B | `qwen72b` | Dense GQA | 72B | varies by GPU |

### Workloads

| ID | Name | Description | Priority | Context Range |
|----|------|-------------|----------|---------------|
| WL-1 | `chat` | High-concurrency multi-turn conversational serving | Throughput | 50-4K input tokens |
| WL-2 | `agent` | Multi-turn agentic tool-calling with growing context | Latency | 1.5K-15K input tokens |
| WL-3 | `rag` | Prefill-dominant with 32K-128K retrieved contexts | TTFT | 32K-128K input tokens |
| WL-4 | `coding` | Repository-context-heavy with high prefix reuse | Balanced | 6K-30K input tokens |

### Modes

| Mode | Name | Description |
|------|------|-------------|
| Mode A | Default vLLM | Stock vLLM with auto-configured settings. Baseline for comparison. |
| Mode B | InferScope Optimized | vLLM with InferScope-tuned engine parameters. |
| Mode C | Operator Submitted | Third-party operator configurations. See [CONTRIBUTING.md](docs/CONTRIBUTING.md). |

### Quantizations

| Format | Support | Usage |
|--------|---------|-------|
| `fp8` | Default for all cells | Primary evaluation quantization |
| `bf16` | Reference baseline | 1 trial per cell for quality/correctness reference |
| `nvfp4` | B200, B300 only | Additional evaluation on supported hardware |

---

## CLI Reference

ISB-1 provides a unified CLI accessible via the `isb1` command.

### `isb1 validate`

Validate configuration files before running benchmarks.

```bash
isb1 validate --sweep configs/sweep/core.yaml
isb1 validate --sweep configs/sweep/core.yaml --config-root configs/
isb1 validate --all-yaml --config-root configs/
```

**Options:**
- `--sweep PATH` — Path to the sweep configuration YAML to validate.
- `--config-root PATH` — Root directory for config files.
- `--all-yaml` — Parse-check every YAML file under the config root.

Validation includes:
- YAML parse integrity
- Required keys in GPU, model, and workload configs
- GPU quantization format support
- Memory fit estimation (model size vs. available HBM)
- Minimum GPU count requirements

### `isb1 plan`

Display the execution plan for a sweep without running anything.

```bash
isb1 plan --config configs/sweep/core.yaml
```

**Options:**
- `--config PATH` / `--sweep PATH` — Path to the sweep configuration YAML.
- writes the plan to stdout

### `isb1 run`

Execute a full benchmark sweep across the configured matrix.

```bash
isb1 run --config configs/sweep/core.yaml --output results/
isb1 run --config configs/sweep/extended.yaml --output results/ --dry-run
```

**Options:**
- `--config PATH` / `--sweep PATH` — Path to the sweep configuration YAML.
- `--output PATH` — Output directory for results.
- `--dry-run` — Print the execution plan without running.
- `--resume` — Resume a previously interrupted sweep.

### `isb1 run-cell`

Execute a single benchmark cell (one GPU x model x workload x mode combination).

```bash
isb1 run-cell \
  --gpu h100 \
  --model llama70b \
  --workload chat \
  --mode mode_a \
  --quantization fp8 \
  --output results/
```

**Options:**
- `--gpu NAME` — GPU short name (for example `h100`, `h200`, `b200`, `b300`).
- `--model NAME` — Model short name (for example `llama70b`, `dsr1`).
- `--workload NAME` — Workload name (`chat`, `agent`, `rag`, `coding`).
- `--mode NAME` — Execution mode (`mode_a`, `mode_b`, `mode_c`).
- `--quantization NAME` — Quantization format (default: `fp8`).
- `--trial N` — Trial number.
- `--output PATH` — Output directory for results.

### `isb1 analyze`

Aggregate completed runs and compute metrics.

```bash
isb1 analyze --results-dir results/ --output analysis.json
```

**Options:**
- `--results-dir PATH` / `--input PATH` — Directory containing completed benchmark runs.
- `--output PATH` — JSON file for consolidated analysis output.

### `isb1 claims`

Review whether completed runs satisfy the publication prerequisites for benchmark claims.

```bash
isb1 claims --results-dir results/
```

**Options:**
- `--results-dir PATH` / `--input PATH` — Directory containing completed benchmark runs.

### `isb1 leaderboard`

Display a ranked leaderboard from a consolidated analysis JSON file.

```bash
isb1 leaderboard --analysis analysis.json
```

**Options:**
- `--analysis PATH` / `--input PATH` — Path to the analysis JSON file.
- `--sort-by METRIC` — Primary ranking metric (default: `generation_throughput`).
- `--top N` — Number of entries to display.

### `isb1 report`

Render an HTML report from a consolidated analysis JSON file.

```bash
isb1 report --analysis analysis.json --output report.html
```

**Options:**
- `--analysis PATH` / `--input PATH` — Path to the analysis JSON file.
- `--output PATH` — Output HTML report path.
- `--template PATH` — Optional custom Jinja2 template.

---

## Configuration Reference

ISB-1 uses a hierarchical YAML configuration system under `configs/`.

### GPU Configuration (`configs/gpus/*.yaml`)

```yaml
gpu_name: "H100-SXM-80GB"       # Full hardware name
gpu_short: "h100"                # Short identifier used in CLI and results
compute_capability: "sm_90a"     # CUDA compute capability
hbm_capacity_gb: 80             # HBM capacity in GB
hbm_bandwidth_tbs: 3.35         # HBM bandwidth in TB/s
fp_formats: ["bf16", "fp16", "fp8_e4m3", "fp8_e5m2"]
nvfp4_support: false            # Whether nvfp4 quantization is supported
kv_cache_dtypes: ["auto", "fp8_e5m2"]
max_gpus_per_node: 8
```

### Model Configuration (`configs/models/*.yaml`)

```yaml
model_name: "Llama-3.3-70B"
model_short: "llama70b"
hf_model_id: "meta-llama/Llama-3.3-70B-Instruct"
architecture_class: "dense_gqa"
total_params_b: 70
min_gpus:                        # Minimum GPUs required per quantization/GPU combo
  bf16:
    h100: 2
    h200: 1
  fp8:
    h100: 1
    h200: 1
recommended_topology:
  fp8:
    h100: "tp1"
```

### Workload Configuration (`configs/workloads/*.yaml`)

```yaml
workload_name: "chat"
workload_id: "wl1"
description: "High-concurrency, short-context, multi-turn conversational serving"
context:
  input_tokens:
    median: 500
    min: 50
    max: 4000
  output_tokens:
    median: 200
    min: 10
    max: 1024
arrival:
  model: "poisson"
  rate_sweep: [1, 2, 4, 8, 16, 32, 64, 128, 256]
slo:
  ttft_p95_ms: 2000
  tpot_p95_ms: 100
```

### Sweep Configuration (`configs/sweep/*.yaml`)

```yaml
sweep_name: "core"
gpus: ["h100", "h200", "b200"]
models:
  - model: "dsr1"
  - model: "llama70b"
workloads: ["chat", "agent", "rag", "coding"]
modes: ["mode_a", "mode_b"]
quantizations:
  default: ["fp8"]
  bf16_reference: true
trials:
  default: 3
  high_variance_max: 5
measurement:
  warmup_requests: 100
  measurement_duration_seconds: 600
variance:
  cv_threshold: 0.10
```

---

## Results Format

### Raw Results

Each benchmark run produces a JSON file in `results/raw/` containing per-request latency data alongside a `manifest.json` with run metadata.

### Aggregated Metrics

After analysis, each cell produces a `CellMetrics` record with the following fields:

| Category | Metrics |
|----------|---------|
| **TTFT** | `ttft_p50`, `ttft_p95`, `ttft_p99` (seconds) |
| **TPOT** | `tpot_p50`, `tpot_p95`, `tpot_p99` (seconds) |
| **ITL** | `itl_p50`, `itl_p95`, `itl_p99` (seconds) |
| **E2E** | `e2e_p50`, `e2e_p95`, `e2e_p99` (seconds) |
| **Throughput** | `generation_throughput` (tok/s), `request_throughput` (req/s) |
| **Goodput** | `goodput` (good req/s), `slo_attainment` (fraction) |
| **Engine** | `kv_cache_utilization_p50/p95`, `preemptions_per_minute`, `queue_depth_p50/p95` |
| **Cache** | `prefix_cache_hit_rate` (fraction) |
| **Power** | `avg_power_watts`, `watts_per_token` |
| **Errors** | `error_rate` (fraction) |

**Critical invariant:** TTFT is excluded from TPOT and ITL calculations. TPOT is computed as `(e2e_latency - ttft) / (output_tokens - 1)`. ITL gaps are computed between consecutive tokens starting from token index 2, thereby excluding the TTFT interval.

### Lockfiles

Each run produces a lockfile in `lockfiles/` containing:
- vLLM version and git hash
- CUDA and PyTorch versions
- Full `nvidia-smi -q` output and NVLink topology
- Complete `pip freeze` package list
- SHA-256 hashes of all config files used
- Random seeds

---

## Contributing

We welcome contributions from the community. See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) for full details.

### Mode C Submissions

Hardware vendors and infrastructure operators can submit Mode C configurations to be included in the benchmark. Submissions must include:

1. A complete vLLM engine argument set as a YAML file.
2. The specific GPU, model, and workload combinations the config targets.
3. Evidence that the config has been tested and produces valid results.

### Code Contributions

1. Fork the repository.
2. Create a feature branch.
3. Ensure all tests pass: `make test`
4. Ensure code style compliance: `ruff check . && black --check .`
5. Submit a pull request with a clear description of changes.

---

## License

ISB-1 is released under the [Apache License, Version 2.0](LICENSE).

```
Copyright 2025 InferScope

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0
```
