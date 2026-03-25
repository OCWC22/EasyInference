# EasyInference

EasyInference is the umbrella repository for two production-facing deliverables:

- **ISB-1** — a reproducible benchmark standard and execution harness for LLM inference serving in `products/isb1/`
- **InferScope** — a self-contained CLI + MCP for inference optimization, diagnostics, and benchmark planning in `products/inferscope/`

The benchmark and the optimizer live in the same repository, but they remain intentionally separate products. That keeps the benchmark neutral and reproducible while keeping InferScope self-contained for operators, open-source users, and enterprise deployments.

## Repository layout

```text
EasyInference/
├── products/
│   ├── isb1/         # Benchmark standard, harness, configs, analysis, tests
│   └── inferscope/   # CLI, MCP server, optimization engine, packaged eval assets
├── .github/workflows/
├── docs/             # Legacy benchmark doc redirects
├── ARCHITECTURE.md
├── CONTRIBUTING.md
├── VALIDATION.md
└── Makefile          # Monorepo delegator
```

## Choose the right product

### ISB-1 benchmark

Use `products/isb1/` if you need to:
- run repeatable serving benchmarks
- validate hardware / model / workload configurations
- produce benchmark reports, leaderboards, and claims
- compare engine behavior under standardized workloads

Start here:
- [ISB-1 product README](products/isb1/README.md)
- [ISB-1 architecture](products/isb1/docs/ARCHITECTURE.md)
- [ISB-1 quickstart](products/isb1/docs/QUICKSTART.md)

### InferScope

Use `products/inferscope/` if you need to:
- recommend serving configurations for vLLM, SGLang, TRT-LLM, Dynamo, or ATOM
- expose optimization and diagnostics through MCP
- audit live endpoints and telemetry
- run InferScope’s packaged benchmark planning and replay tools

Start here:
- [InferScope product README](products/inferscope/README.md)
- [InferScope architecture](products/inferscope/ARCHITECTURE.md)
- [InferScope benchmark subsystem docs](products/inferscope/docs/BENCHMARKS.md)

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

- **Separate packaging:** each product keeps its own `pyproject.toml`
- **Separate CI:** benchmark and InferScope validate in distinct GitHub Actions workflows
- **Separate runtime state:** ISB-1 writes into `products/isb1/results/` by default; InferScope writes benchmark artifacts to `~/.inferscope/benchmarks/` by default
- **No forced shared runtime library:** shared repository does not mean shared product internals

## Contributing and validation

- Monorepo guide: [CONTRIBUTING.md](CONTRIBUTING.md)
- Validation guide: [VALIDATION.md](VALIDATION.md)
- Root Make targets: `make validate`, `make isb1-lint`, `make isb1-format-check`, `make test`, `make inferscope-lint`, `make inferscope-package-smoke`, and `make all-checks`

## License

This repository contains multiple licenses:
- `products/isb1/` — Apache-2.0
- `products/inferscope/` — MIT

See the root [LICENSE](LICENSE) notice and each product-local license file.
