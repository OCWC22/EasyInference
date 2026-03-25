# Validation Reports

InferScope validation reports are versioned snapshots.

Within the EasyInference monorepo, this product lives at `products/inferscope/`.

## Naming convention

Use:

```text
NN-YYYY-MM-DD-name.md
```

## Latest reports

- [01-2026-03-23-ai-first-validation.md](validations/01-2026-03-23-ai-first-validation.md)
- [02-2026-03-23-benchmark-and-stress-test-plan.md](validations/02-2026-03-23-benchmark-and-stress-test-plan.md)

## Current validation contract

As of **March 24, 2026**, the repo should be considered valid when the following checks pass:

- `uv run ruff check ...`
- `uv run ruff format --check ...`
- `uv run mypy src/inferscope/`
- `uv run bandit -r src/inferscope/ -c pyproject.toml -ll`
- package-data smoke checks from a built wheel for packaged benchmark workloads and experiment specs

Optional validation layers:

- `uv run pytest ...` when a `tests/` tree is present in the checkout
- live integration tests when the repo contains that suite and the required endpoint secrets are configured

## Notes

- Treat each file in `validations/` as a point-in-time report, not a permanent statement about the current repo.
- If docs drift from code, trust the current package layout and executable CLI/MCP surfaces.
- TRT-LLM and Dynamo compiler surfaces exist, but live-validation breadth should still be documented conservatively.
