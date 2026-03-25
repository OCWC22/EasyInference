# InferScope Benchmark Plan

InferScope's benchmark plan exists to answer one question:

> Do InferScope's recommendations measurably improve inference behavior for real workloads on real hardware?

## Benchmark boundary

Built-in workloads and experiment specs are owned by the packaged evaluation subsystem under `src/inferscope/benchmarks/`. The source tree no longer treats a repo-root `benchmarks/` directory as authoritative.

This makes the benchmark subsystem:

- self-contained in the installable package
- consistent across CLI and MCP usage
- easier to extract into a dedicated repo later if needed

## Core benchmark surfaces

- `benchmark-workloads` — list packaged workload built-ins
- `benchmark-experiments` — list packaged experiment built-ins
- `benchmark-plan` — resolve a concrete run plan
- `benchmark` — replay a workload against an OpenAI-compatible endpoint
- `benchmark-compare` — compare saved artifacts
- `benchmark-stack-plan` / `benchmark-stack-write` — model full experiment stacks

## Workload classes to keep shipping

- coding long-context
- enterprise tool-agent
- medical and legal RAG
- mixed neo-cloud traffic
- chat and reasoning workloads

## Measurement priorities

- TTFT and end-to-end latency
- throughput and queue depth
- KV cache pressure and preemptions
- prefix cache hit rate
- before/after artifact comparison for rollout review

## Validation philosophy

Each claim InferScope makes should be testable with:

1. a baseline engine configuration
2. an InferScope-generated configuration
3. a workload pack that reflects a real usage pattern
4. saved artifacts plus metrics snapshots for later review

## Future extraction goal

If the benchmark subsystem becomes its own repo later, the extraction target should preserve the current contract:

- packaged built-ins remain authoritative
- core optimization remains an inward dependency
- runtime artifacts stay outside the source tree
- CLI and MCP call patterns stay stable
