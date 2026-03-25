# InferScope Benchmarks

InferScope ships a packaged benchmark subsystem for operators.

This is distinct from ISB-1:

- **ISB-1** owns the benchmark standard and canonical workload families.
- **InferScope** owns practical built-ins, replay workflows, and artifact handling for CLI and MCP use.

## Built-in benchmark assets

InferScope ships packaged resources under:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

These built-ins are available from both the CLI and MCP server.

## Core commands

```bash
# discover built-ins
inferscope benchmark-workloads
inferscope benchmark-experiments

# inspect a concrete run plan
inferscope benchmark-plan tool-agent http://localhost:8000

# replay a workload against an endpoint
inferscope benchmark coding-long-context http://localhost:8000

# compare two saved artifacts
inferscope benchmark-compare before.json after.json

# materialize a deployment stack bundle
inferscope benchmark-stack-plan vllm-single-endpoint-baseline h100
```

## Procedural built-ins

Some built-ins can be procedurally expanded at runtime.

Current support:

- `tool-agent`
- `coding-long-context`

Supported procedural options:

- `--synthetic-requests`
- `--synthetic-input-tokens`
- `--synthetic-output-tokens`
- `--synthetic-seed`
- `--context-file` for `coding-long-context`

Example:

```bash
inferscope benchmark-plan tool-agent http://localhost:8000 \
  --synthetic-requests 8 \
  --synthetic-input-tokens 4096 \
  --synthetic-output-tokens 512
```

```bash
inferscope benchmark coding-long-context http://localhost:8000 \
  --synthetic-requests 4 \
  --synthetic-input-tokens 32768 \
  --synthetic-output-tokens 768 \
  --context-file ./repo_context.txt
```

These commands still resolve to a standard `WorkloadPack` and write a normal `BenchmarkArtifact`.

## Relationship to ISB-1

InferScope built-ins should map back to the stable ISB-1 families rather than inventing a second benchmark taxonomy.

Current mapping:

- `tool-agent` → ISB-1 `agent`
- `coding-long-context` → ISB-1 `coding`

This lets the MCP surface move quickly without destabilizing the benchmark standard.

## Legacy compatibility

Legacy repo-style references continue to resolve for packaged built-ins, for example:

- `benchmarks/workloads/coding-long-context.yaml`
- `benchmarks/experiment_specs/vllm-single-endpoint-baseline.yaml`

Use the short built-in names for new automation and docs.

## Artifact location

InferScope benchmark artifacts default to:

```text
~/.inferscope/benchmarks/
```

This keeps runtime output out of the source tree and makes the MCP behavior match the CLI behavior.
