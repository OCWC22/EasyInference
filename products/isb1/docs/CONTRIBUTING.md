# Contributing to ISB-1

Thank you for your interest in contributing to the ISB-1 benchmark. This document covers how operators submit Mode C configurations, the format requirements, the validation process, and general code contribution guidelines.

---

## Table of Contents

- [Mode C Configuration Submissions](#mode-c-configuration-submissions)
- [Submission Format](#submission-format)
- [Validation Process](#validation-process)
- [Submission Guidelines](#submission-guidelines)
- [Code Contributions](#code-contributions)
- [Reporting Issues](#reporting-issues)

---

## Mode C Configuration Submissions

Mode C allows hardware vendors, cloud providers, and infrastructure operators to submit their optimized vLLM serving configurations for evaluation under the ISB-1 standard. Mode C results are benchmarked using the same methodology, measurement infrastructure, and statistical rigor as Modes A and B.

### Who Can Submit

- GPU hardware vendors
- Cloud inference platform operators
- vLLM plugin or extension developers
- Independent optimization researchers

### What You Submit

A Mode C submission is a set of YAML configuration files that define vLLM engine arguments for specific cells in the benchmark matrix. Each configuration targets a particular (GPU, model, workload) combination.

---

## Submission Format

### Directory Structure

```
submissions/<operator_name>/
├── metadata.yaml          # Required: operator info and submission scope
├── configs/
│   ├── <gpu>_<model>_<workload>.yaml   # One per targeted cell
│   └── ...
└── notes.md               # Optional: explanation of optimization strategy
```

### metadata.yaml

```yaml
operator_name: "Acme Inference Co"
operator_contact: "benchmark-team@acme.example.com"
submission_date: "2025-12-01"
description: "Optimized configurations for H100 and B200 targeting DeepSeek-R1 and Llama-3.3-70B."

# Which cells this submission covers
targets:
  - gpu: "h100"
    model: "dsr1"
    workloads: ["chat", "agent", "rag", "coding"]
  - gpu: "h100"
    model: "llama70b"
    workloads: ["chat", "agent"]
  - gpu: "b200"
    model: "dsr1"
    workloads: ["chat", "agent", "rag", "coding"]

# vLLM version this config was developed and tested against
vllm_version: "0.8.x"
cuda_version: "12.8"

# Optional: link to public documentation or blog post
reference_url: "https://acme.example.com/inference-benchmark-2025"
```

### Per-Cell Configuration Files

Each YAML file defines the complete vLLM engine arguments for one cell:

```yaml
# configs/h100_dsr1_chat.yaml

cell:
  gpu: "h100"
  model: "dsr1"
  workload: "chat"
  quantization: "fp8"

engine_args:
  model: "deepseek-ai/DeepSeek-R1"
  tensor-parallel-size: 8
  max-model-len: 32768
  quantization: "fp8"
  kv-cache-dtype: "fp8_e5m2"
  gpu-memory-utilization: 0.92
  enable-chunked-prefill: true
  max-num-batched-tokens: 65536
  max-num-seqs: 512
  enable-prefix-caching: true
  disable-log-requests: true
  # Add any additional vLLM engine arguments here

# Optional: document why these values were chosen
rationale: |
  Increased gpu-memory-utilization to 0.92 to maximize KV cache capacity.
  Enabled chunked prefill with 65536 max batched tokens to balance
  prefill and decode throughput on H100 8-GPU configuration.
```

### Requirements for Configuration Files

1. **Complete engine arguments.** Every engine parameter that deviates from the vLLM default must be explicitly listed. Do not rely on implicit defaults.
2. **No external dependencies.** Configurations must not require custom vLLM builds, plugins, or patches that are not publicly available.
3. **Reproducible.** The configuration must produce consistent results when run multiple times with the same benchmark version.
4. **Valid quantization.** The quantization format must be supported by the target GPU (the config validator will check this).
5. **Memory feasible.** The model must fit in the target GPU's HBM at the specified configuration (the config validator will estimate this).

---

## Validation Process

### Automated Validation

All submissions are validated using the ISB-1 config validator:

```bash
# Validate a single cell config
isb1 validate --config-root configs/ --sweep submissions/acme/configs/h100_dsr1_chat.yaml

# The automated CI will also run:
python -m harness.config_validator --all-yaml --config-root submissions/acme/
```

The validator checks:
- YAML parse integrity
- Required keys present in metadata and cell configs
- GPU quantization support
- Approximate memory fit (model size vs. HBM with KV cache overhead)
- Minimum GPU count met

### Manual Review

After automated validation, submissions undergo manual review:

1. **Scope check.** The submission targets valid cells in the benchmark matrix.
2. **No prohibited modifications.** The configuration must not disable safety checks, bypass measurement instrumentation, or alter the workload traces.
3. **Engine argument review.** All engine arguments are reviewed for correctness and absence of parameters that would compromise measurement integrity.
4. **Test run.** A subset of the submitted cells are run to verify the configuration starts successfully and produces valid output.

### Acceptance Criteria

A submission is accepted if:
- It passes all automated validation checks.
- It passes manual review.
- The test run completes without errors.
- The output quality evaluation does not show degradation beyond tolerance.

### Rejection Reasons

Common reasons for rejection:
- Configuration requires a custom or unreleased vLLM build.
- Engine arguments would compromise measurement integrity (e.g., disabling warmup or altering the request rate).
- Model does not fit in the specified GPU configuration.
- Incomplete or missing metadata.

---

## Submission Guidelines

### How to Submit

1. Fork the repository.
2. Create a directory under `submissions/<your_operator_name>/`.
3. Add your `metadata.yaml`, cell config files, and optional `notes.md`.
4. Run the config validator locally to verify your configs.
5. Submit a pull request with the title: `[Mode C] <Operator Name> submission for <GPU(s)>`.
6. The ISB-1 maintainers will review, validate, and merge if accepted.

### Timing

Mode C submissions are accepted on a rolling basis. Results from accepted submissions are included in the next benchmark publication cycle.

### Updates

Operators may update their submissions at any time by opening a new pull request. Updated configurations replace previous ones for the same cells. The benchmark history retains all prior results for comparison.

### Confidentiality

All submitted configurations are public. Do not include proprietary parameters, internal URLs, or credentials in your submission. If your optimization requires a proprietary component, contact the maintainers to discuss options.

---

## Code Contributions

### Development Setup

```bash
git clone <repository-url>
cd EasyInference/products/isb1
pip install -e ".[dev]"
```

### Code Style

- Python 3.10+ syntax.
- Formatting: `black` with `line-length = 100`.
- Linting: `ruff` with `line-length = 100`, target `py310`.
- Type hints on all public function signatures.

### Running Tests

```bash
make test
# or
python -m pytest tests/ -v
```

### Pull Request Process

1. Create a feature branch from `main`.
2. Make your changes.
3. Ensure all tests pass: `make test`
4. Ensure code style: `ruff check . && black --check .`
5. Write a clear PR description explaining the motivation and changes.
6. Request review from a maintainer.

### What We Are Looking For

- Bug fixes with regression tests.
- New workload generators that cover additional production patterns.
- Analysis and visualization improvements.
- Documentation improvements.
- Performance improvements to the harness itself.

---

## Reporting Issues

If you encounter a bug, measurement inconsistency, or documentation error, please open an issue with:

1. A clear title describing the problem.
2. Steps to reproduce.
3. Expected vs. actual behavior.
4. Relevant configuration files and lockfile contents.
5. System information (GPU model, vLLM version, CUDA version).
