# InferScope Architecture

InferScope is a hardware-aware operator product with two public surfaces:

- the `inferscope` CLI
- the `inferscope serve` MCP server

Both surfaces expose the same benchmark-aware optimization and runtime profiling core.

## NVIDIA recommendation boundary

InferScope now uses a shared platform-policy layer to keep NVIDIA recommendation behavior consistent across:

- the recommendation DAG
- engine selection ranking
- engine compilers
- validator warnings
- memory and KV overflow advisories
- benchmark stack planning

That layer makes the following distinctions explicit instead of inferring them indirectly:

- **H100 vs H200**
- **B200/B300 vs GB200/GB300**
- supported auto-selected engines vs preview planning targets
- HBM fit vs Grace coherent overflow advisory

## Boundary model

InferScope sits on top of the EasyInference benchmark stack.

- **InferenceX** is the external public reference.
- **ISB-1** is the reproducible benchmark standard.
- **InferScope** is the operator layer that packages benchmark assets and runtime analysis for direct use.

This means InferScope should not grow into a second benchmark standard. It should stay focused on planning, replay, comparison, diagnostics, and profiling.

## Repository layout

```text
src/inferscope/
├── optimization/         # recommendation DAG, checks, serving profiles
├── engines/              # engine compiler + runtime adapter seam
├── hardware/             # GPU metadata and detection
├── models/               # model metadata
├── telemetry/            # Prometheus scraping, normalization, shared snapshots
├── profiling/            # runtime profiling core and future trace/kernel boundary
├── tools/                # operator-facing audits, diagnostics, profiling wrappers
├── benchmarks/           # packaged workloads, experiments, replay, artifacts
├── cli.py                # primary CLI composition root
├── cli_profiling.py      # runtime profiling CLI commands
├── cli_benchmarks.py     # benchmark CLI commands
├── server.py             # primary MCP composition root
├── server_profiling.py   # runtime profiling MCP tools
└── server_benchmarks.py  # benchmark MCP tools
```

## Dependency direction

- optimization does not depend on benchmark orchestration
- telemetry owns shared runtime snapshot models and capture helpers
- profiling depends on telemetry, optimization checks, and engine adapters
- benchmarks depend on telemetry capture and artifact models, not the other way around
- CLI and MCP surfaces compose the same profiling and benchmark subsystems

## Runtime profiling subsystem

`src/inferscope/profiling/` now owns the first production runtime profiling path.

Current flow:

1. scrape Prometheus metrics from a live endpoint
2. normalize engine-specific metrics into a shared runtime shape
3. classify workload heuristically
4. run audit checks against a `DeploymentContext`
5. group findings into a stable bottleneck taxonomy
6. optionally preview tuning changes
7. optionally enrich runtime identity from `/v1/models`

The profiling core is deliberately isolated so future `nsys`, `rocprofv3`, or kernel-level integrations can land in the same package without leaking into CLI/MCP composition or benchmark orchestration.

This matters for Hopper and Blackwell specifically because the profiling surface now shares the same platform vocabulary as the optimizer. The MCP can reason about what the deployment is supposed to do and what the runtime metrics say it is actually doing.

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

Benchmark experiment metadata now has enough structure to describe realistic cache tiers and overflow layers, including:

- `gpu_hbm`
- `grace_coherent`
- `cpu_dram`
- `remote_cache`

That lets InferScope describe operator studies like:

- single-endpoint cold-session offload with `OffloadingConnector`
- disaggregated prefill/decode with `LMCacheConnectorV1`
- Grace-aware long-context overflow on GH200 / GB200 / GB300 systems

## MCP bridge workloads

Two built-ins currently act as the main bridge from benchmark methodology into operator workflows:

- `tool-agent`
- `coding-long-context`

They are intentionally practical rather than normative. They map back to the stable ISB-1 families rather than redefining the benchmark taxonomy.
