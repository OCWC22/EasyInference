# InferScope

InferScope is the operator-facing product in EasyInference: a self-contained CLI and MCP for inference tuning, diagnostics, planning, and benchmark replay.

## Product role

EasyInference has three distinct layers of concern:

- **InferenceX** is the external public reference for market-wide benchmark results.
- **ISB-1** is the reproducible benchmark standard in this repo.
- **InferScope** is the high-leverage operator surface that turns those ideas into day-to-day workflows.

If the goal is benchmarking through an MCP, this is the product that matters.

## What InferScope does

InferScope helps operators:

- choose an engine and serving profile
- validate topology, quantization, and memory fit
- audit running endpoints
- plan benchmark stacks
- replay benchmark workloads against OpenAI-compatible endpoints
- compare saved benchmark artifacts before and after a change

## Benchmark bridge

InferScope packages benchmark assets under `src/inferscope/benchmarks/` and exposes them through both the CLI and MCP server.

Key bridge workloads:

- `tool-agent` — MCP / tool-calling benchmark pack
- `coding-long-context` — long-context repository review and coding benchmark pack

These are practical operator-facing built-ins, not new benchmark-standard families. They map back to the canonical ISB-1 families:

- `tool-agent` → `agent`
- `coding-long-context` → `coding`

The local `inferscope-bench/` tree informed these workload shapes, but it is not a public product or runtime dependency.

## Procedural benchmark materialization

InferScope can expand certain built-in workloads procedurally at runtime.

Current supported procedural built-ins:

- `tool-agent`
- `coding-long-context`

CLI example:

```bash
inferscope benchmark-plan tool-agent http://localhost:8000 \
  --synthetic-requests 8 \
  --synthetic-input-tokens 4096 \
  --synthetic-output-tokens 512 \
  --synthetic-seed 7
```

Long-context coding with an external context file:

```bash
inferscope benchmark coding-long-context http://localhost:8000 \
  --synthetic-requests 4 \
  --synthetic-input-tokens 32768 \
  --synthetic-output-tokens 768 \
  --context-file ./repo_context.txt
```

The resulting benchmark plan still resolves to the same stable `WorkloadPack` and `BenchmarkArtifact` contracts.

## Quick start

```bash
cd EasyInference/products/inferscope
uv sync --dev

# recommendation and validation
inferscope recommend DeepSeek-R1 h100 --num-gpus 8 --workload coding
inferscope validate Llama-3-70B h200 --tp 2 --quantization fp8

# benchmark catalog and replay
inferscope benchmark-workloads
inferscope benchmark-plan tool-agent http://localhost:8000 --synthetic-requests 4
inferscope benchmark coding-long-context http://localhost:8000 --synthetic-requests 2

# MCP server
inferscope serve
```

## Runtime storage

InferScope keeps runtime benchmark output outside the source tree by default:

```text
~/.inferscope/benchmarks/
```

This directory holds benchmark artifacts, comparisons, and generated stack bundles.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/BENCHMARK-PLAN.md](docs/BENCHMARK-PLAN.md)
- [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md)
- [VALIDATION.md](VALIDATION.md)

## License

MIT
