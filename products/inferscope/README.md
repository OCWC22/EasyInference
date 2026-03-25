# InferScope

InferScope is the operator-facing product in EasyInference: a self-contained CLI and MCP for inference tuning, runtime profiling, diagnostics, planning, and benchmark replay.

## Product role

EasyInference has three distinct layers of concern:

- **InferenceX** is the external public reference for market-wide benchmark results.
- **ISB-1** is the reproducible benchmark standard in this repo.
- **InferScope** is the high-leverage operator surface that turns those ideas into day-to-day workflows.

If the goal is benchmarking and profiling through an MCP, this is the product that matters.

## NVIDIA platform stance

As of **March 25, 2026**, InferScope's production recommendation path is validated around:

- **H100**
- **H200**
- **B200**
- **GB200**

Key policy decisions:

- **vLLM** and **SGLang** are the supported NVIDIA auto-selection paths
- **TRT-LLM** and **Dynamo** remain **preview planning targets**
- Blackwell is treated explicitly as **B200/B300 vs GB200/GB300**, not inferred from raw memory size
- Grace coherent overflow is surfaced as an advisory capacity tier, not silently treated as normal HBM fit

This is intentionally complementary to InferenceX: InferenceX measures the public frontier continuously; InferScope turns those platform assumptions into operator recommendations, validation, profiling, and benchmark workflows.

## What InferScope does

InferScope helps operators:

- choose an engine and serving profile
- validate topology, quantization, and memory fit
- profile live endpoints through Prometheus-based runtime analysis
- audit bottlenecks and preview tuning changes
- plan benchmark stacks
- replay benchmark workloads against OpenAI-compatible endpoints
- compare saved benchmark artifacts before and after a change

## Benchmark and MCP focus

The highest-leverage InferScope loop is:

1. recommend a serving plan
2. validate it against platform/model constraints
3. benchmark it against a representative workload
4. profile the live endpoint through MCP
5. tighten the deployment from observed bottlenecks

That is the main reason this product exists.

## Runtime profiling

InferScope now exposes a first-class runtime profiling surface:

- CLI: `inferscope profile-runtime http://localhost:8000`
- MCP: `tool_profile_runtime`

This surface is Prometheus-first:

- scrape `/metrics`
- normalize cross-engine runtime metrics
- classify workload heuristically
- run audit checks
- group findings into bottlenecks
- optionally preview scheduler/cache tuning
- optionally enrich runtime identity from `/v1/models`

`inferscope profile` remains the static model-intel command. `profile-runtime` is the live endpoint profiler.

## Benchmark bridge

InferScope packages benchmark assets under `src/inferscope/benchmarks/` and exposes them through both the CLI and MCP server.

Key bridge workloads:

- `tool-agent` — MCP / tool-calling benchmark pack
- `coding-long-context` — long-context repository review and coding benchmark pack
- `long-context-kv-offload-rag` — realistic multi-turn long-context RAG pack for KV spill and cold-session reuse

These are practical operator-facing built-ins, not new benchmark-standard families. They map back to the canonical ISB-1 families:

- `tool-agent` → `agent`
- `coding-long-context` → `coding`
- `long-context-kv-offload-rag` → `rag`

The local `inferscope-bench/` tree informed these workload shapes, but it is not a public product or runtime dependency.

InferScope also exposes a structured benchmark matrix catalog so operators can filter packaged workloads and experiment lanes by:

- target GPU family
- target model class
- workload class
- focus area
- engine

Primary surfaces:

- CLI: `inferscope benchmark-matrix`
- MCP: `tool_get_benchmark_matrix`

InferScope also exposes a benchmark-strategy layer that turns a concrete deployment target into:

- the right primary workload
- the right baseline / offload / disaggregated benchmark lanes
- the right live profiling bridge for follow-up tuning

Primary surfaces:

- CLI: `inferscope benchmark-strategy`
- MCP: `tool_plan_benchmark_strategy`

## Benchmark architecture lanes

InferScope now carries three distinct long-context operator lanes:

- **GPU-resident baseline** — compare against a clean single-endpoint vLLM or SGLang path
- **OffloadingConnector lane** — single-endpoint vLLM with explicit cold-session KV spill
- **LMCache disaggregated lane** — vLLM prefill/decode split with LMCache and optional Grace-aware overflow modeling

These lanes are the main way InferScope extends beyond InferenceX today: not by cloning the public leaderboard, but by giving operators a reproducible way to study realistic KV-tiering behavior.

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

# runtime profiling and audit
inferscope profile-runtime http://localhost:8000 --gpu-arch sm_90a
inferscope audit http://localhost:8000 --gpu-arch sm_90a

# benchmark catalog and replay
inferscope benchmark-workloads
inferscope benchmark-matrix --focus-area kv_offload --gpu-family blackwell_grace
inferscope benchmark-strategy Qwen3.5-72B gb200 --workload long_context_rag --num-gpus 4
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

Runtime profiles are **not** persisted to disk by default in v1.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md)
- [docs/PROFILING.md](docs/PROFILING.md)
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
- [docs/BENCHMARK-PLAN.md](docs/BENCHMARK-PLAN.md)
- [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md)
- [VALIDATION.md](VALIDATION.md)

## License

MIT
