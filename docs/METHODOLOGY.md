# ISB-1 Benchmark Methodology

This document defines the complete measurement methodology for the ISB-1 (InferScope Benchmark Standard, Version 1) benchmark. All results published under the ISB-1 standard must adhere to the procedures described here.

---

## Table of Contents

- [Benchmark Design Principles](#benchmark-design-principles)
- [Workload Definitions](#workload-definitions)
- [Measurement Methodology](#measurement-methodology)
- [Metric Definitions](#metric-definitions)
- [Mode Definitions](#mode-definitions)
- [Statistical Methodology](#statistical-methodology)
- [Quality Evaluation Track](#quality-evaluation-track)
- [Claim Evaluation Framework](#claim-evaluation-framework)
- [Reproducibility Requirements](#reproducibility-requirements)

---

## Benchmark Design Principles

ISB-1 is designed around five core principles:

1. **Production fidelity.** Workloads must represent real-world inference serving patterns. Synthetic traces use realistic vocabulary, conversation structures, and context-length distributions rather than lorem ipsum or uniform random data.

2. **Measurement rigor.** Every reported metric must be backed by statistically sufficient data. Warmup periods must be validated. Steady-state must be confirmed. Variance must be quantified and bounded.

3. **Reproducibility.** Every run must produce a lockfile capturing the complete software, hardware, and configuration state. A third party must be able to reproduce any reported result within statistical tolerance using the lockfile and trace data.

4. **Fairness across modes.** The benchmark defines explicit modes (A, B, C) to separate baseline engine behavior from optimized configurations. Comparisons are always made within the same hardware and workload context.

5. **Transparency.** All methodology choices, thresholds, and definitions are documented and version-controlled. No hidden tuning or unreported configuration changes.

---

## Workload Definitions

ISB-1 defines four workloads that collectively cover the dominant LLM inference serving patterns in production.

### WL-1: Chat (Multi-Turn Conversational Serving)

- **Config:** `configs/workloads/chat.yaml`
- **Workload ID:** `wl1`
- **Description:** High-concurrency, short-context, multi-turn conversational serving.
- **Trace source:** ShareGPT V3 dataset (filtered to 1-5 user turns) with synthetic fallback using a curated English vocabulary bank.
- **Context distribution:**
  - Input tokens: 50-4,000 (median 500)
  - Output tokens: 10-1,024 (median 200)
- **Arrival model:** Poisson process
- **Rate sweep:** 1, 2, 4, 8, 16, 32, 64, 128, 256 req/s
- **Session behavior:** Sticky sessions with prefix reuse enabled.
- **Sampling:** temperature=0.7, top_p=0.9, max_tokens=1024
- **SLO gates:** TTFT p95 <= 2,000 ms, TPOT p95 <= 100 ms
- **Priority:** Throughput
- **Saturation detection:** TTFT p99 > 10,000 ms or error rate > 1%

### WL-2: Agent (Multi-Turn Tool-Calling)

- **Config:** `configs/workloads/agent.yaml`
- **Workload ID:** `wl2`
- **Description:** Multi-turn agentic workload with structured tool-calling and growing context.
- **Trace source:** Synthetic agent conversations with 5 tool schemas (search, code_execute, file_read, database_query, api_call).
- **Context distribution:**
  - Input tokens: 1,500-15,000 (linear growth per turn)
  - Output tokens: 100-1,500 (median 400)
- **Arrival model:** Gamma distribution (shape=2) for bursty arrival patterns typical of agent workloads.
- **Rate sweep:** 0.5, 1, 2, 4, 8, 16, 32, 64 req/s
- **Session behavior:** Sticky sessions with prefix reuse (shared tool-schema prefix of ~2,000 tokens).
- **Sampling:** temperature=0.0 (greedy for tool-calling determinism), max_tokens=2048
- **SLO gates:** TTFT p95 <= 1,500 ms, TPOT p95 <= 80 ms
- **Priority:** Latency
- **Conversation structure:** 3-8 turns per session following the pattern: user query -> model tool_call -> tool result -> model continuation.

### WL-3: RAG (Retrieval-Augmented Generation)

- **Config:** `configs/workloads/rag.yaml`
- **Workload ID:** `wl3`
- **Description:** Prefill-dominant workload with 32K-128K token retrieved contexts.
- **Trace source:** Synthetic RAG traces with 5-20 retrieval chunks of 1,000-5,000 tokens each.
- **Context distribution:**
  - Input tokens: Bimodal distribution
    - Cluster 1: center 32,000 tokens (60% weight)
    - Cluster 2: center 96,000 tokens (40% weight)
  - Output tokens: 100-800 (median 400)
- **Arrival model:** Poisson process
- **Rate sweep:** 0.25, 0.5, 1, 2, 4, 8, 16 req/s
- **Session behavior:** Non-sticky, no prefix reuse between sessions (but shared system prompt prefix of ~500 tokens).
- **Sampling:** temperature=0.0, max_tokens=1024
- **SLO gates:** TTFT p95 <= 6,000 ms (32K context), <= 20,000 ms (96K context); TPOT p95 <= 100 ms
- **Priority:** TTFT

### WL-4: Coding (Repository-Context Assistance)

- **Config:** `configs/workloads/coding.yaml`
- **Workload ID:** `wl4`
- **Description:** Repository-context-heavy coding workload with high prefix reuse.
- **Trace source:** Synthetic coding traces with 5-15 files of 200-2,000 tokens each, 0-10 conversation turns.
- **Context distribution:**
  - Input tokens: 6,000-30,000 (accumulative growth per turn)
  - Output tokens: 200-2,000 (median 500)
- **Arrival model:** Poisson process
- **Rate sweep:** 1, 2, 4, 8, 16, 32, 64, 128 req/s
- **Session behavior:** Sticky sessions with prefix reuse (shared system prompt + repo context of ~10,000 tokens).
- **Sampling:** temperature=0.0, max_tokens=4096
- **SLO gates:** TTFT p95 <= 3,000 ms, TPOT p95 <= 60 ms
- **Priority:** Balanced

---

## Measurement Methodology

### Warmup Phase

Before any measurement data is collected, each benchmark cell undergoes a warmup phase:

1. **Warmup requests:** 100 requests are sent before measurement begins.
2. **Warmup duration:** Minimum 60 seconds of warmup traffic.
3. **Steady-state validation:** The warmup validator monitors throughput variance across sliding windows. Steady-state is declared when the coefficient of variation of throughput across consecutive windows falls below the `steady_state_variance_threshold` (default: 0.20).
4. **Warmup extensions:** If steady-state is not reached, the warmup period is extended up to 3 times (`warmup_max_extensions`). If steady-state is never achieved, the cell is flagged as `"unstable"` and the run proceeds with a warning.

### Steady-State Measurement

Once warmup completes:

1. **Measurement duration:** 600 seconds (10 minutes) of steady-state measurement per rate point.
2. **Rate sweep:** Each workload defines a list of request rates. The benchmark iterates through each rate point sequentially, running the full measurement duration at each rate.
3. **Saturation detection:** If TTFT p99 exceeds the configured threshold or the error rate exceeds 1%, the sweep stops at that rate (the system has reached saturation).

### Cooldown

A 30-second cooldown period separates consecutive rate points to allow the engine to drain queues and stabilize.

### Trial Repetition

- **Default trials:** 3 trials per cell.
- **High variance:** If the coefficient of variation across trials exceeds the `cv_threshold` (10%), additional trials are run up to `high_variance_max` (5 trials).
- **bf16 reference:** 1 trial only (used for quality comparison, not performance ranking).

---

## Metric Definitions

### Time To First Token (TTFT)

The elapsed time from when the request is sent to when the first output token is received by the client. Measured in seconds.

- Reported percentiles: p50, p95, p99.
- Reflects the prefill latency of the model.

### Time Per Output Token (TPOT)

**Critical: TTFT is excluded from TPOT.**

TPOT measures the average per-token decode latency, computed as:

```
TPOT = (e2e_latency - ttft) / (output_tokens - 1)
```

By subtracting TTFT from the end-to-end latency, TPOT isolates the decode phase and is not inflated by prefill time. Requests with fewer than 2 output tokens are excluded from TPOT calculations.

- Reported percentiles: p50, p95, p99.
- Measured in seconds per token.

### Inter-Token Latency (ITL)

**Critical: TTFT is excluded from ITL.**

ITL measures the gap between consecutive token arrivals, starting from the second output token onward (token index 2). The gap between the request start and the first token (the TTFT interval) is explicitly excluded.

```
ITL gaps = [t[i] - t[i-1] for i in range(2, len(token_timestamps))]
```

- Reported percentiles: p50, p95, p99.
- Measured in seconds.

### End-to-End Latency (E2E)

Total time from request submission to completion of the final output token. Includes prefill, decode, and any queuing delay.

- Reported percentiles: p50, p95, p99.
- Measured in seconds.

### Generation Throughput

Total output tokens generated across all concurrent requests divided by the wall-clock measurement duration.

```
generation_throughput = total_output_tokens / wall_clock_seconds
```

- Measured in tokens per second (tok/s).

### Request Throughput

Number of successfully completed requests divided by the wall-clock measurement duration.

```
request_throughput = successful_requests / wall_clock_seconds
```

- Measured in requests per second (req/s).

### Goodput

The rate of requests that meet all SLO constraints, measured per wall-clock second. A request is "good" if and only if:

1. Its TTFT is at or below the configured TTFT SLO threshold (default: 2.0s).
2. Its TPOT is at or below the configured TPOT SLO threshold (default: 0.1s).

```
goodput = count_of_good_requests / wall_clock_seconds
```

### SLO Attainment

The fraction of successful requests that meet all SLO constraints.

```
slo_attainment = count_of_good_requests / successful_requests
```

### Engine-Level Metrics

| Metric | Description |
|--------|-------------|
| `kv_cache_utilization_p50/p95` | KV cache memory utilization (0-1) |
| `prefix_cache_hit_rate` | Fraction of prefix cache hits (0-1) |
| `preemptions_per_minute` | Rate of request preemptions |
| `queue_depth_p50/p95` | Number of requests queued for scheduling |

### Power Metrics

| Metric | Description |
|--------|-------------|
| `avg_power_watts` | Mean GPU power draw during measurement |
| `watts_per_token` | Power efficiency: avg_power / generation_throughput |

---

## Mode Definitions

### Mode A: Default vLLM

Mode A uses the stock vLLM inference engine with auto-configured settings. The only parameters explicitly set are those required by the benchmark harness:

- Model path / HuggingFace ID
- Quantization format (fp8, bf16, or nvfp4)
- Tensor parallelism degree (from `recommended_topology`)
- Port and logging configuration

All other engine parameters (scheduling policy, chunked prefill, speculative decoding, etc.) use vLLM defaults. Mode A results establish the baseline against which all optimizations are measured.

**Purpose:** Provide a fair, reproducible baseline that any user would achieve with a standard vLLM deployment.

### Mode B: InferScope Optimized

Mode B uses vLLM with InferScope-tuned engine parameters. These may include:

- Custom scheduling parameters
- Chunked prefill configuration
- Speculative decoding settings
- KV cache management tuning
- Prefix caching configuration
- Batch size limits and memory allocation ratios

The complete Mode B configuration for each cell is recorded in the lockfile. Mode B results demonstrate the performance achievable with expert-level engine tuning.

**Purpose:** Establish optimized performance baselines and quantify the benefit of engine tuning.

### Mode C: Operator Submitted

Mode C accepts configurations submitted by third-party hardware vendors, cloud providers, or infrastructure operators. See [CONTRIBUTING.md](../docs/CONTRIBUTING.md) for submission requirements.

Mode C configurations:
- Must pass the config validator (`isb1 validate`).
- Must specify the exact GPU, model, and workload combinations they target.
- Must be fully reproducible (all engine arguments documented).
- Are run under the same measurement methodology as Modes A and B.

**Purpose:** Enable vendors to demonstrate their optimized serving configurations under standardized, independent measurement.

---

## Statistical Methodology

### Paired T-Tests

Mode comparisons (e.g., Mode A vs. Mode B) use paired two-sided t-tests on matched trial observations. This accounts for inter-trial variability by pairing measurements from the same trial index.

- **Null hypothesis:** No difference in the metric between modes.
- **Significance level:** alpha = 0.05.
- **Output:** t-statistic, p-value, mean difference, and 95% confidence interval on the mean difference.
- **Requirement:** At least 2 paired observations (in practice, 3 or more trials).

### Bootstrap Confidence Intervals

For reporting metric uncertainty, ISB-1 uses BCa (bias-corrected and accelerated) bootstrap confidence intervals:

- **Bootstrap resamples:** 10,000.
- **Confidence level:** 95%.
- **Bias correction:** Computed from the proportion of bootstrap statistics below the point estimate.
- **Acceleration:** Estimated via jackknife influence values.

Bootstrap CIs are reported for all primary metrics (throughput, latency percentiles, goodput).

### Coefficient of Variation (CV) Thresholds

The coefficient of variation (standard deviation divided by the mean) is used to assess measurement stability:

- **CV threshold:** 10% (`cv_threshold: 0.10`)
- If the CV of a metric across trials exceeds 10%, the cell is flagged as high-variance.
- High-variance cells are automatically extended to 5 trials.
- If CV remains above 10% after 5 trials, the cell is reported with a high-variance warning.

### Trial Sufficiency

The `needs_more_trials` function checks whether the CV exceeds the threshold and returns a boolean. The sweep orchestrator uses this to decide whether to run additional trials for a cell.

---

## Quality Evaluation Track

Performance optimizations must not degrade output quality. ISB-1 includes a quality evaluation track that runs independently of the throughput/latency benchmark.

### ROUGE-L Evaluation

- Reference outputs are generated using the bf16 reference configuration (Mode A, bf16 quantization, single trial).
- Test outputs from each cell (optimized quantization and mode) are compared against the reference using ROUGE-L scoring.
- Configuration: `configs/quality/rouge.yaml`

### HumanEval

- Code generation accuracy is tested using the HumanEval benchmark.
- Each model is evaluated through the serving endpoint to measure pass@1 under the deployed configuration.
- Configuration: `configs/quality/humaneval.yaml`

### MMLU-Pro

- General knowledge and reasoning quality is assessed using MMLU-Pro.
- Configuration: `configs/quality/mmlu_pro.yaml`

### RULER

- Long-context retrieval accuracy is tested using the RULER benchmark, which measures the model's ability to retrieve information from extended contexts.
- Configuration: `configs/quality/ruler.yaml`

### Quality Pass Criteria

A cell passes quality evaluation if:
1. ROUGE-L score against bf16 reference is above the configured threshold.
2. HumanEval pass@1 does not degrade by more than the configured tolerance compared to the bf16 reference.
3. MMLU-Pro accuracy is within the configured tolerance of the bf16 reference.

---

## Claim Evaluation Framework

ISB-1 includes a claim evaluation framework for independently validating vendor performance assertions.

### Claim Structure

A claim consists of:
- **Subject:** The configuration being evaluated (GPU, model, workload, mode).
- **Assertion:** A quantitative performance statement (e.g., "Mode B achieves 2x throughput over Mode A on WL-1 with llama70b on H100").
- **Metric:** The specific metric the claim references.
- **Threshold:** The claimed improvement factor or absolute value.

### Evaluation Process

1. The claim evaluator loads aggregated metrics for the relevant cells.
2. It computes the actual delta or ratio between the claimed configurations.
3. It runs a paired t-test to determine if the difference is statistically significant.
4. It produces a claim report with:
   - Measured values for both configurations
   - Computed delta/ratio
   - Statistical significance (p-value)
   - Bootstrap 95% CI on the difference
   - Verdict: **Supported**, **Not Supported**, or **Inconclusive**

### Verdicts

| Verdict | Criteria |
|---------|----------|
| **Supported** | Measured improvement meets or exceeds the claim, and the difference is statistically significant (p < 0.05). |
| **Not Supported** | Measured improvement does not meet the claim, or the difference is not statistically significant. |
| **Inconclusive** | Insufficient data or high variance prevents a definitive conclusion. |

---

## Reproducibility Requirements

Every ISB-1 benchmark run must satisfy the following reproducibility requirements:

### Lockfile

Each run produces a JSON lockfile capturing:
- **vLLM:** Version string and git commit hash.
- **CUDA:** Version string from `nvcc --version` or `torch.version.cuda`.
- **PyTorch:** Version string.
- **nvidia-smi:** Full `nvidia-smi -q` output (GPU model, driver version, memory, clocks, ECC status).
- **NVLink topology:** `nvidia-smi topo -m` output.
- **System:** `uname -a` output.
- **Python packages:** Complete `pip freeze` listing.
- **Config hashes:** SHA-256 of every YAML config file used.
- **Model revision:** HuggingFace commit SHA for the model weights.
- **Random seeds:** All seeds used for trace generation and sampling.

### Trace Determinism

Workload traces are generated with fixed seeds and saved as JSONL files. The same seed and configuration must produce identical traces.

### Configuration Integrity

The SHA-256 hash of each config file is recorded in the lockfile. Any modification to configs between runs is detectable.

### Third-Party Reproduction

A third party must be able to reproduce any reported result by:
1. Checking out the benchmark code at the reported version.
2. Using the lockfile to install matching software versions.
3. Using the recorded config hashes to verify config integrity.
4. Running the same sweep or cell with the same seeds.

Results are considered reproduced if the measured metrics fall within the bootstrap 95% CI of the original run.
