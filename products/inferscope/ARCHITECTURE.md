# InferScope Architecture

InferScope is a hardware-aware operator product with two public surfaces:

- the `inferscope` CLI
- the `inferscope serve` MCP server

Both surfaces expose the same benchmark-aware optimization core.

## Boundary model

InferScope sits on top of the EasyInference benchmark stack.

- **InferenceX** is the external public reference.
- **ISB-1** is the reproducible benchmark standard.
- **InferScope** is the operator layer that packages benchmark assets for direct use.

This means InferScope should not grow into a second benchmark standard. It should stay focused on planning, replay, comparison, and diagnostics.

## Repository layout

```text
src/inferscope/
├── optimization/        # recommendation DAG and serving profiles
├── engines/             # engine compiler seam
├── hardware/            # GPU metadata and detection
├── models/              # model metadata
├── tools/               # operator-facing audits and diagnostics
├── benchmarks/          # packaged workloads, experiments, replay, artifacts
├── profiling/           # future profiler/kernel boundary
├── cli.py               # primary CLI composition root
├── cli_benchmarks.py    # benchmark CLI commands
├── server.py            # primary MCP composition root
└── server_benchmarks.py # benchmark MCP tools
```

## Dependency direction

- optimization does not depend on benchmark orchestration
- benchmarks may depend inward on optimization and engine metadata
- CLI and MCP surfaces compose the same underlying benchmark subsystem
- profiling remains advisory-only today

## Benchmark subsystem

`src/inferscope/benchmarks/` is the source of truth for built-in benchmark assets.

It owns:

- packaged workload YAMLs
- packaged experiment specs
- workload catalog resolution
- OpenAI-compatible replay
- benchmark artifact persistence
- procedural materialization for selected built-ins

The key bridge API is `materialize_workload(...)`.

Behavior:

- built-in workloads can be loaded as static seed packs
- selected built-ins can be procedurally expanded at runtime
- explicit file paths still work, but procedural expansion is limited to packaged built-ins

## MCP bridge workloads

Two built-ins currently act as the main bridge from benchmark methodology into operator workflows:

- `tool-agent`
- `coding-long-context`

They are intentionally practical rather than normative. They map back to the stable ISB-1 families rather than redefining the benchmark taxonomy.

## Profiling boundary

`src/inferscope/profiling/` exists so future profiler or kernel work can evolve without coupling itself to the replay and MCP surfaces.

Today it records advisory profiling intent. Later it can host profiler execution helpers, trace export, and kernel-facing integrations.
