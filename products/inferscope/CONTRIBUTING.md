# Contributing to InferScope

InferScope is the operator-facing CLI and MCP product in EasyInference.

Use this product when your change affects:

- recommendation and validation flows
- operator diagnostics and audits
- packaged benchmark workloads or experiment specs
- benchmark replay through the CLI or MCP server
- benchmark artifact handling or comparison

## Development setup

```bash
git clone <repository-url>
cd EasyInference/products/inferscope
uv sync --dev
```

## Required local checks

```bash
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/inferscope/
uv run pytest tests/ -v --tb=short
```

If your change affects package security posture, also run:

```bash
uv run bandit -r src/inferscope/ -c pyproject.toml -ll
```

## Benchmark contribution rules

Packaged benchmark assets belong only under:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

Do not introduce repo-root mirrors of packaged built-ins.

If you modify the benchmark bridge, verify at minimum:

```bash
uv run inferscope benchmark-workloads
uv run inferscope benchmark-experiments
uv run inferscope benchmark-matrix --workload-class tool_agent --engine sglang
uv run inferscope benchmark-plan tool-agent http://localhost:8000 --synthetic-requests 2 || true
```

The `tool-agent` and `coding-long-context` built-ins are the main MCP bridge workloads. Keep them mapped to stable ISB-1 families rather than creating a second benchmark taxonomy.

## Pull request expectations

1. keep changes scoped
2. update docs when CLI or MCP behavior changes
3. add tests when benchmark plan resolution, replay contracts, or artifact structure changes
4. call out rollout implications in the PR description
