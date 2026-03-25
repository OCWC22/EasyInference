# EasyInference

EasyInference is a two-product monorepo for inference benchmarking and operator tooling:

- **ISB-1** — the **Inference Serving Benchmark Standard 1** in `products/isb1/`
- **InferScope** — the **CLI + MCP** for inference optimization, diagnostics, and benchmark replay in `products/inferscope/`

## Benchmark ecosystem position

EasyInference is designed to be **complementary** to [InferenceX](https://inferencex.semianalysis.com/), not a clone of it.

- **InferenceX** is the external, continuously updated, vendor-neutral public reference for cross-hardware and cross-framework inference performance.
- **ISB-1** is the reproducible benchmark standard for local validation, methodology, publication, and scenario-specific workloads.
- **InferScope** is the operator surface that exposes recommendation, diagnostics, profiling, and benchmark tooling through a CLI and MCP.

As of **March 25, 2026**, the InferScope NVIDIA recommendation path is explicitly hardened around the same hardware families InferenceX publicly tracks today: **H100, H200, B200, GB200, and GB300**. EasyInference does not try to replace that public leaderboard; it tries to make those platform choices actionable for operators.

The extension path is explicit:

- InferenceX covers the public cross-vendor frontier
- EasyInference extends that with operator-facing scenarios such as:
  - long-context coding
  - tool-agent / MCP workloads
  - realistic KV-cache overflow and cold-session reuse
  - LMCache / disaggregated-prefill studies
  - Grace-coherent overflow modeling for GH200 / GB200 / GB300 systems

This repo also absorbs workload ideas from the local `inferscope-bench/` donor harness — especially **MCP/tool-call** and **long-context coding** patterns — without turning that subtree into a third public product.

## Repository layout

```text
EasyInference/
├── products/
│   ├── isb1/         # Benchmark standard, harness, configs, analysis, tests
│   └── inferscope/   # CLI, MCP server, optimization engine, packaged benchmark tooling
├── inferscope-bench/ # Local donor harness used as a benchmark foundation, not a public product
├── .github/workflows/
├── docs/
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── VALIDATION.md
└── Makefile
```

## Choose the right product

### ISB-1 benchmark

Use `products/isb1/` if you need to:
- run reproducible serving benchmarks against canonical workload families
- validate hardware / model / workload configurations
- publish benchmark reports and claims
- test scenario coverage such as chat, agent, RAG, and coding

ISB-1 now executes its own generated traces through an internal OpenAI-compatible replay path. That keeps the benchmark aligned with its canonical workload definitions instead of depending on an external synthetic runner.

Start here:
- [ISB-1 product README](products/isb1/README.md)
- [ISB-1 architecture](products/isb1/docs/ARCHITECTURE.md)
- [ISB-1 methodology](products/isb1/docs/METHODOLOGY.md)
- [ISB-1 ecosystem positioning](products/isb1/docs/ECOSYSTEM.md)

### InferScope

Use `products/inferscope/` if you need to:
- recommend serving configs for vLLM, SGLang, or ATOM, with preview planning for TRT-LLM and Dynamo
- expose optimization and diagnostics through MCP
- replay packaged benchmark workloads against a real endpoint
- procedurally expand **tool-agent** and **coding-long-context** workloads from the benchmark bridge
- materialize benchmark stacks for cache-aware or disaggregated serving experiments

Start here:
- [InferScope product README](products/inferscope/README.md)
- [InferScope architecture](products/inferscope/ARCHITECTURE.md)
- [InferScope benchmark docs](products/inferscope/docs/BENCHMARKS.md)

## Quick start from the monorepo root

### Benchmark

```bash
pip install -e "./products/isb1[dev,quality]"
make isb1-validate
```

### InferScope

```bash
cd products/inferscope
uv sync --dev
uv run inferscope --help
```

## Operational stance

- **Two products only:** ISB-1 and InferScope remain the supported public surfaces
- **Separate packaging:** each product keeps its own `pyproject.toml`
- **Separate CI:** benchmark and InferScope validate in distinct workflows
- **Separate runtime state:**
  - ISB-1 writes benchmark outputs under `products/isb1/results/`
  - InferScope writes artifacts under `~/.inferscope/benchmarks/`
- **No shared runtime library:** shared repo does not imply merged product internals

## Contributing and validation

- Monorepo guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Validation guide: [VALIDATION.md](VALIDATION.md)
- Root Make targets: `make validate`, `make isb1-lint`, `make isb1-format-check`, `make test`, `make inferscope-lint`, `make inferscope-package-smoke`, and `make all-checks`

## License

This repository contains multiple licenses:
- `products/isb1/` — Apache-2.0
- `products/inferscope/` — MIT

See the root [LICENSE](LICENSE) notice and each product-local license file.
