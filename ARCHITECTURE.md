# EasyInference Monorepo Architecture

EasyInference is a two-product monorepo built around one operating model:

- **ISB-1** is the reproducible benchmark standard.
- **InferScope** is the operator-facing CLI and MCP layer.

The products share a repository for release discipline and discoverability. They do not share a runtime package or a single merged codebase.

## External reference vs local products

EasyInference is designed to be **complementary** to [InferenceX](https://inferencex.semianalysis.com/), not a dashboard clone.

- **InferenceX** is the public, continuously updated, market-wide reference for cross-hardware and cross-framework inference performance.
- **ISB-1** is the reproducible benchmark standard inside this repo for controlled validation, publication, and operator review.
- **InferScope** is the high-leverage operator product: a self-contained MCP and CLI for tuning, diagnostics, planning, and benchmark replay.

The local `inferscope-bench/` tree is treated as a **donor foundation** for workload ideas and replay patterns. It is not a third product and should not become a public dependency.

## Product boundaries

### `products/isb1/`

ISB-1 owns the benchmark standard:

- `workloads/` — canonical workload generators and trace materialization
- `harness/` — server lifecycle, replay execution, telemetry, manifests, lockfiles
- `analysis/` — metric computation, aggregation, statistics, and reporting helpers
- `quality/` — quality checks that sit beside performance results
- `configs/`, `publication/`, `scripts/`, and `tests/`

ISB-1 keeps its own Python namespaces (`harness`, `workloads`, `analysis`, `quality`) inside the product root. That preserves the `isb1` CLI and keeps benchmark-local workflows self-contained.

### `products/inferscope/`

InferScope owns the operator product:

- `src/inferscope/optimization/` — serving profile and recommendation DAG
- `src/inferscope/engines/` — engine compiler seam
- `src/inferscope/hardware/` and `src/inferscope/models/` — hardware and model metadata
- `src/inferscope/tools/` — operator-facing diagnostics and audits
- `src/inferscope/benchmarks/` — packaged workloads, experiments, replay, and artifact handling
- `src/inferscope/profiling/` — future profiler/kernel boundary

InferScope depends inward on its benchmark package. The benchmark package may depend inward on the optimizer. The optimizer must not depend on benchmark orchestration.

## Benchmark-to-MCP bridge

The key repository-level bridge is:

1. **ISB-1** defines the neutral workload families and replay methodology.
2. **InferScope** packages practical, operator-facing workload packs and experiment specs.
3. InferScope exposes those benchmark assets through both the CLI and MCP server.
4. Operator flows can move from recommendation → replay → artifact comparison without leaving the product.

Current mapping:

- `tool-agent` in InferScope maps into the ISB-1 **agent** family.
- `coding-long-context` in InferScope maps into the ISB-1 **coding** family.
- RAG and chat scenarios remain available as neutral benchmark families in ISB-1 and as packaged evaluation assets in InferScope.

## Root ownership

The repository root owns only monorepo surfaces:

- `README.md` — landing page and positioning
- `ARCHITECTURE.md` — product boundaries and bridge model
- `CONTRIBUTING.md` — contribution routing
- `VALIDATION.md` — validation index
- `Makefile` — delegating entrypoints
- `.github/workflows/` — per-product CI
- `docs/` — root-level compatibility and index material

## Runtime data model

- **ISB-1** defaults to product-local repo storage under `products/isb1/`.
- **InferScope** defaults to user-local runtime storage under `~/.inferscope/`.

This is intentional. Benchmark publications stay repo-oriented; operator artifacts stay package-oriented.
