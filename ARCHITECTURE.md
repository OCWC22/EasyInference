# EasyInference Monorepo Architecture

EasyInference is organized as a two-product monorepo.

## Product boundaries

### `products/isb1/`
The benchmark product owns:
- `harness/` — execution lifecycle, server orchestration, telemetry, lockfiles
- `analysis/` — metrics, statistics, aggregation, reporting plots
- `quality/` — evaluation logic
- `workloads/` — workload generators and schemas
- `configs/`, `results/`, `publication/`, `scripts/`, and `tests/`

Key point: the benchmark keeps its original Python namespaces (`harness`, `analysis`, `quality`, `workloads`) inside the product root. That preserves its CLI surface (`isb1`) and config references without forcing a broader rename.

### `products/inferscope/`
The InferScope product owns:
- `src/inferscope/` — the packaged CLI and MCP server
- `optimization/`, `engines/`, `hardware/`, `models/`, and `tools/` — the optimization core
- `src/inferscope/benchmarks/` — packaged evaluation assets and planning logic
- `src/inferscope/profiling/` — the future profiler/kernel seam

Key point: InferScope remains self-contained. Its internal benchmark package is not the same thing as the standalone ISB-1 product.

## Deliberate non-goals

This repository does **not** introduce:
- a shared runtime library consumed by both products
- a single root Python package
- a forced merge of benchmark and optimization internals

The repository is shared for organization, discoverability, and release discipline. The products stay operationally independent.

## Root-level ownership

The repository root owns only monorepo surfaces:
- `README.md` — product selection and landing page
- `CONTRIBUTING.md` — contribution routing
- `VALIDATION.md` — validation index
- `Makefile` — delegating entrypoints
- `.github/workflows/` — per-product CI
- `docs/` — compatibility redirects for moved ISB-1 docs

## Runtime data model

- **ISB-1:** defaults to product-local filesystem roots under `products/isb1/`
- **InferScope:** defaults to user-local cache/runtime roots under `~/.inferscope/`

That separation is intentional. Benchmark outputs remain repo-oriented; InferScope remains package-oriented.
