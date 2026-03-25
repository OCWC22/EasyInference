# InferScope Architecture

InferScope is a hardware-aware inference optimization product with a small number of explicit subsystem boundaries.

## Repository boundaries

```text
src/inferscope/
├── optimization/   # normalized ServingProfile DAG and tuning logic
├── engines/        # engine compiler seam (vLLM, SGLang, ATOM, TRT-LLM, Dynamo)
├── hardware/       # GPU profiles and hardware detection
├── models/         # model metadata and sizing rules
├── tools/          # CLI/MCP-facing operational tools
├── benchmarks/     # packaged evaluation subsystem
├── profiling/      # advisory seam for future profiler/kernel work
├── cli.py          # user-facing CLI composition root
├── cli_benchmarks.py
├── server.py       # MCP composition root
└── server_benchmarks.py
```

The dependency direction is intentional:

- **core optimization does not depend on benchmarks**
- **benchmarks depend inward on core optimization and engine compilers**
- **profiling is advisory-only today** and exists so future kernel work does not need to land inside the core DAG file

## DAG pipeline

InferScope's recommendation flow is orchestrated in `optimization/recommender.py`:

```text
HardwareNode → ModelNode → WorkloadNode → ProfilingNode → TelemetryNode → CompilerNode
```

### Node responsibilities

1. **HardwareNode** selects an engine family and precision path from GPU ISA capabilities.
2. **ModelNode** resolves topology boundaries such as TP/DP/EP and speculative decoding.
3. **WorkloadNode** maps workload shape to scheduler, cache, offload, and chunked-prefill policy.
4. **ProfilingNode** records an advisory profiling intent (`nsys` or `rocprofv3`) via the profiling subsystem.
5. **TelemetryNode** appends operator-facing metrics and alert guidance.
6. **CompilerNode** binds the normalized `ServingProfile` to an engine-specific command surface.

## Public surfaces

### CLI

`src/inferscope/cli.py` owns the main Typer app. Benchmark and evaluation commands are registered from `src/inferscope/cli_benchmarks.py`, keeping the core operator commands separate from evaluation orchestration.

### MCP server

`src/inferscope/server.py` owns the main FastMCP server. Benchmark and artifact tools are registered from `src/inferscope/server_benchmarks.py`, which keeps evaluation features isolated without changing public tool names.

## Evaluation subsystem

`src/inferscope/benchmarks/` is the single source of truth for built-in workloads and experiment specs.

- `catalog.py` resolves packaged built-ins and supports legacy repo-path aliases for compatibility.
- `experiments.py` and `launchers.py` generate concrete run plans and benchmark stack bundles.
- `openai_replay.py` runs workloads against OpenAI-compatible endpoints and writes artifacts under `~/.inferscope/benchmarks/`.

This makes the installed package self-contained while leaving the benchmark subsystem easy to extract into a separate repo later if needed.

## Profiling boundary

`src/inferscope/profiling/` currently contains advisory intent resolution only. That package exists to host future integrations such as:

- `nsys` and `rocprofv3` execution helpers
- kernel advisory metadata
- trace export policies
- profiler-specific validation or artifact export

The important architectural point is that future kernel work now has a home that does not force deeper coupling with benchmark orchestration or the core optimizer.
