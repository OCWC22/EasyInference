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
- [03-2026-03-25-runtime-profiling-v1.md](validations/03-2026-03-25-runtime-profiling-v1.md)
- [04-2026-03-25-hopper-blackwell-hardening.md](validations/04-2026-03-25-hopper-blackwell-hardening.md)

## Current validation contract

As of **March 25, 2026**, InferScope should be considered valid when the following checks pass:

- `uv run ruff check src/ tests/`
- `uv run ruff format --check src/ tests/`
- `uv run mypy src/inferscope/`
- `uv run pytest tests/ -v --tb=short`
- `uv run bandit -r src/inferscope/ -c pyproject.toml -ll`
- built-wheel smoke checks for packaged benchmark workloads and experiment specs
- built-wheel smoke checks for procedural benchmark plan resolution on `tool-agent` and `coding-long-context`
- runtime profiling unit coverage for telemetry capture, profiling orchestration, CLI registration, and MCP-safe wrapper behavior
- Hopper/Blackwell recommendation coverage for **H100**, **H200**, **B200**, and **GB200**
- compiler regression coverage for explicit **B200 vs GB200** separation
- benchmark launcher coverage showing benchmark plans inherit the same H200/B200/GB200 policy as the MCP
- benchmark launcher coverage for explicit **OffloadingConnector** and **LMCache + Grace** long-context lanes
- benchmark catalog coverage for packaged descriptor metadata and filtered matrix generation across CLI, MCP, and Python entrypoints
- benchmark strategy coverage for benchmark-suite planning and runtime-bridge prioritization

## Notes

- Treat files in `validations/` as point-in-time reports, not permanent guarantees.
- If docs drift from code, trust the current packaged CLI and MCP surfaces.
- Procedural benchmark generation is limited to selected packaged built-ins, not arbitrary YAML file paths.
- Runtime profiling is Prometheus-first in v1 and does not persist profiles to disk by default.
- TRT-LLM and Dynamo are still preview planning targets in InferScope even when InferenceX publishes results for them.
