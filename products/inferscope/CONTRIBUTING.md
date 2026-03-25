# Contributing to InferScope

Thank you for your interest in contributing.

## Development setup

```bash
git clone https://github.com/your-org/EasyInference.git
cd EasyInference/products/inferscope
uv sync --dev
```

## Local quality checks

```bash
# Lint + format
uv run ruff check src/
uv run ruff format --check src/

# Type checking
uv run mypy src/inferscope/

# Security scan
uv run bandit -r src/inferscope/ -c pyproject.toml
```

If a `tests/` tree exists in your checkout, also run:

```bash
uv run pytest tests/ -v
```

## Benchmark contribution rules

Built-in benchmark assets belong only in the packaged evaluation subsystem:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

Do **not** add repo-root mirror copies of packaged workloads or experiment specs.

When modifying packaged benchmark assets, at minimum verify:

```bash
uv run inferscope benchmark-workloads
uv run inferscope benchmark-experiments
```

## Pull request process

1. Branch from `main`
2. Keep changes scoped and reviewable
3. Update docs when public behavior changes
4. Run the required validation checks locally
5. Submit a PR with a clear summary and rollout implications

## Security

- Never commit credentials, API keys, or secrets
- All HTTP requests must use the validated URL helper
- All user inputs must be validated before use
- See `SECURITY.md` for the full security policy
