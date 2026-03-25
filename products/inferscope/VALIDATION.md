# Validation Reports

InferScope validation reports are versioned snapshots.

Within the EasyInference monorepo, this product lives at `products/inferscope/`.

## Naming convention

```text
NN-YYYY-MM-DD-name.md
```

## Latest reports

- [01-2026-03-23-ai-first-validation.md](validations/01-2026-03-23-ai-first-validation.md)
- [02-2026-03-23-benchmark-and-stress-test-plan.md](validations/02-2026-03-23-benchmark-and-stress-test-plan.md)

## Current validation contract

As of **March 25, 2026**, InferScope should be considered valid when the following checks pass:

- `uv run ruff check src/ tests/`
- `uv run ruff format --check src/ tests/`
- `uv run mypy src/inferscope/`
- `uv run pytest tests/ -v --tb=short`
- `uv run bandit -r src/inferscope/ -c pyproject.toml -ll`
- built-wheel smoke checks for packaged benchmark workloads and experiment specs
- built-wheel smoke checks for procedural benchmark plan resolution on `tool-agent` and `coding-long-context`

## Notes

- Treat files in `validations/` as point-in-time reports, not permanent guarantees.
- If docs drift from code, trust the current packaged CLI and MCP surfaces.
- Procedural benchmark generation is limited to selected packaged built-ins, not arbitrary YAML file paths.
