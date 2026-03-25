# ISB-1 Quick Start Guide

Get from zero to your first benchmark result in 5 minutes.

---

## Prerequisites

Before you begin, ensure you have:

- **Python 3.10+** installed
- **NVIDIA GPU** with CUDA 12.x and a compatible driver
- **vLLM** installed and able to serve models
- **HuggingFace access** for gated models (set `HF_TOKEN` environment variable)
- At least one supported GPU: H100 SXM, H200 SXM, B200, or B300

---

## Step 1: Install

```bash
git clone <repository-url>
cd EasyInference

# Install the benchmark package with development and quality extras
pip install -e ".[dev,quality]"
```

Verify the CLI is available:

```bash
isb1 --help
```

---

## Step 2: Setup

Run the automated setup to install telemetry tooling, download datasets, and generate workload traces:

```bash
make setup
```

This executes:
1. `scripts/setup_node.sh` -- system-level prerequisites
2. `scripts/install_telemetry.sh` -- GPU telemetry collectors
3. `scripts/download_datasets.sh` -- ShareGPT and evaluation datasets
4. `scripts/generate_mode_a_configs.py` -- auto-generate Mode A engine configs
5. `scripts/generate_traces.py` -- generate deterministic workload trace files

---

## Step 3: Validate Configuration

Before running a benchmark, validate that your configuration is correct and your hardware can support the requested matrix:

```bash
isb1 validate --sweep configs/sweep/core.yaml
```

This checks:
- All referenced GPU, model, and workload configs exist and parse correctly.
- Models fit in GPU memory at the requested quantization.
- GPUs support the requested quantization formats.
- Minimum GPU count requirements are met.

---

## Step 4: Run a Single Cell

Start with a single cell to verify everything works:

```bash
isb1 run-cell \
  --gpu h100 \
  --model llama70b \
  --workload chat \
  --mode mode_a \
  --quantization fp8 \
  --output results/
```

This will:
1. Start a vLLM server with Llama-3.3-70B at fp8 quantization.
2. Run the warmup phase (100 requests, 60 seconds minimum).
3. Execute the chat workload rate sweep (1 to 256 req/s).
4. Collect GPU telemetry and engine metrics.
5. Save raw results to `results/raw/`.

Expect 15-30 minutes for a single cell depending on workload and hardware.

---

## Step 5: Analyze Results

Aggregate raw results into metrics:

```bash
isb1 analyze --input results/raw --output results/aggregated
```

This computes all ISB-1 metrics: TTFT, TPOT, ITL, throughput, goodput, SLO attainment, and more.

---

## Step 6: View Results

Generate a leaderboard:

```bash
isb1 leaderboard --input results/aggregated --output results/leaderboard
```

Generate publication-ready plots:

```bash
make report
```

Results are written to:
- `results/aggregated/` -- JSON and CSV metric files
- `results/leaderboard/` -- ranked tables
- `publication/figures/` -- throughput-latency curves, leaderboard heatmaps

---

## Running the Full Core Suite

Once you have verified a single cell works, run the full core benchmark:

```bash
# Preview the execution plan
isb1 plan --sweep configs/sweep/core.yaml

# Run all cells
isb1 run --sweep configs/sweep/core.yaml --output results/

# Analyze and generate reports
isb1 analyze --input results/raw --output results/aggregated
isb1 claims --input results/aggregated --output results/claims
isb1 leaderboard --input results/aggregated --output results/leaderboard
make report
```

The core suite covers 3 GPUs, 3 models, 4 workloads, and 2 modes. Expect 24-72 hours of total runtime depending on hardware.

---

## Next Steps

- Read the full [Methodology](METHODOLOGY.md) to understand workload definitions, metric calculations, and statistical methods.
- Read the [Architecture](ARCHITECTURE.md) for a technical deep-dive into the codebase.
- To submit an operator config for Mode C evaluation, see [Contributing](CONTRIBUTING.md).
