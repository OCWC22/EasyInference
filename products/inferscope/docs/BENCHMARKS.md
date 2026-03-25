# InferScope Benchmarks

InferScope ships a packaged evaluation subsystem for workload replay, experiment specs, artifact comparison, and benchmark stack planning.

This is distinct from the standalone ISB-1 benchmark product in `products/isb1/`. InferScope packages its own built-in evaluation assets so the CLI and MCP server remain self-contained.

## Built-ins are packaged resources

Authoritative built-ins live under:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

They ship with the installed package, so the CLI and MCP server can resolve them without relying on repo-relative files.

## Common commands

```bash
# Discover built-ins
inferscope benchmark-workloads
inferscope benchmark-experiments

# Replay a workload by built-in name
inferscope benchmark coding-long-context http://localhost:8000

# Resolve a concrete run plan without executing it
inferscope benchmark-plan reasoning-chatbot http://localhost:8000

# Generate or materialize a benchmark stack bundle
inferscope benchmark-stack-plan vllm-single-endpoint-baseline h100
inferscope benchmark-stack-write vllm-single-endpoint-baseline h100 ./out
```

## Custom files are still supported

You can still pass a custom workload or experiment YAML path to the CLI. If the file exists, InferScope uses it directly.

## Legacy repo-path compatibility

Older commands such as these continue to resolve:

- `benchmarks/workloads/coding-long-context.yaml`
- `benchmarks/experiment_specs/vllm-single-endpoint-baseline.yaml`

They are treated as compatibility aliases for the packaged built-ins. Prefer the shorter built-in names for new docs and automation.

## Artifact and stack locations

Runtime benchmark output defaults to:

```text
~/.inferscope/benchmarks/
```

This directory is used for:

- saved benchmark artifact JSON files
- generated stack bundles under `stacks/`
- local operator output that should not live in the source tree

Override it with `INFERSCOPE_BENCHMARK_DIR` if you need a different location.

## MCP usage

The MCP server exposes benchmark catalog, replay, artifact comparison, and stack materialization tools through the same packaged source of truth. This keeps remote agents and local CLI usage aligned.

## Future extraction boundary

The benchmark subsystem is now isolated enough to split into a dedicated repo later if desired. The important contract today is:

- packaged built-ins remain authoritative
- the public CLI and MCP surfaces remain stable
- benchmark code depends inward on core optimization, not the other way around
