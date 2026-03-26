# InferScope

InferScope is the operator-facing product in EasyInference: a self-contained CLI and MCP for inference tuning, runtime profiling, diagnostics, planning, and benchmark replay.

## Product role

EasyInference has three distinct layers of concern:

- **InferenceX** is the external public reference for market-wide benchmark results.
- **ISB-1** is the reproducible benchmark standard in this repo.
- **InferScope** is the high-leverage operator surface that turns those ideas into day-to-day workflows.

If the goal is benchmarking and profiling through an MCP, this is the product that matters.

## Platform stance

As of **March 25, 2026**, InferScope's production recommendation path is validated around:

**NVIDIA (primary validation path):**
- **H100**, **H200**, **B200**, **GB200**

**AMD (day-one support):**
- **MI300X** (gfx942 / CDNA3)
- **MI355X** (gfx950 / CDNA4)

Key policy decisions:

- **vLLM** and **SGLang** are the supported NVIDIA auto-selection paths
- **vLLM** is the supported AMD auto-selection path
- **TRT-LLM** is a **supported** NVIDIA engine (highest compiled throughput, requires compilation step)
- **Dynamo 1.0** is **recommended** for multi-node NVIDIA disaggregated serving (KV-aware routing, NIXL, SLO autoscaling)
- Blackwell is treated explicitly as **B200/B300 vs GB200/GB300**, not inferred from raw memory size
- Grace coherent overflow is surfaced as an advisory capacity tier, not silently treated as normal HBM fit
- AMD GPUs are supported for planning, benchmark gating, and support assessment; NVIDIA remains the primary validated path

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

InferScope carries distinct long-context operator lanes:

- **GPU-resident baseline** — compare against a clean single-endpoint vLLM or SGLang path
- **OffloadingConnector lane** — single-endpoint vLLM with explicit cold-session KV spill
- **LMCache disaggregated lane** — vLLM prefill/decode split with LMCache and optional Grace-aware overflow modeling
- **Dynamo disaggregated lanes** — Dynamo 1.0 orchestrated prefill/decode split with NIXL KV transfer, KV-aware Smart Router, and SLO-driven autoscaling
- **KV compression lanes** — FP8 KV quantization, prefix caching hit rates, chunked prefill tuning

These lanes are the main way InferScope extends beyond InferenceX today: not by cloning the public leaderboard, but by giving operators a reproducible way to study realistic KV-tiering behavior.

### Dynamo integration

NVIDIA Dynamo 1.0 is the production orchestration layer for disaggregated NVIDIA serving. InferScope treats Dynamo as a first-class engine with orchestration semantics:

- 3 Dynamo experiment specs: coding, RAG, and Grace Blackwell topologies
- Launcher generates Dynamo declarative YAML config, vLLM worker commands, and Smart Router endpoints
- NIXL handles KV cache transfer between prefill/decode pools via RDMA/NVLink
- Strategy auto-selects Dynamo lanes for multi-GPU NVIDIA deployments
- Telemetry scrapes vLLM/SGLang worker endpoints (Dynamo metrics come from the orchestration layer)

Benchmark planning and replay are now also GPU/model/ISA-aware:

- benchmark MCP and CLI surfaces can validate concrete NVIDIA GPUs (**H100/H200/GH200/B200/GB200/GB300**) and AMD GPUs (**MI300X/MI355X**)
- support payloads surface the resolved ISA (NVIDIA: `sm_90a`, `sm_100`, `sm_103`; AMD: `gfx942`, `gfx950`)
- unsupported cache/topology combinations fail early instead of producing misleading plans

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

## Prerequisites

- **Python 3.11+**
- **[uv](https://docs.astral.sh/uv/)** — dependency management and virtual environment
- **Git**
- **NVIDIA GPU + CUDA** or **AMD GPU + ROCm** — only required for live benchmarks/profiling, not for planning or unit tests

## Quick start

```bash
git clone https://github.com/OCWC22/EasyInference.git
cd EasyInference/products/inferscope
uv sync --dev
cp .env.example .env  # optional — see Configuration below

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
inferscope benchmark-plan tool-agent http://localhost:8000 --synthetic-requests 4 --gpu h100 --engine sglang
inferscope benchmark coding-long-context http://localhost:8000 --synthetic-requests 2

# MCP server
inferscope serve
```

## Configuration

InferScope reads optional environment variables from `.env` (gitignored). Copy the example:

```bash
cp .env.example .env
```

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERSCOPE_DEBUG` | `0` | Enable verbose/debug logging |
| `INFERSCOPE_CACHE_DIR` | `~/.inferscope` | Cache directory for empirical profiles and benchmark artifacts |
| `INFERSCOPE_DEFAULT_GPU_UTIL` | `0.92` | Default `gpu_memory_utilization` target for memory planning |

These are optional — InferScope works with sensible defaults out of the box.

## Runtime storage

InferScope keeps runtime benchmark output outside the source tree by default:

```text
~/.inferscope/benchmarks/
```

This directory holds benchmark artifacts, comparisons, and generated stack bundles.

Runtime profiles are **not** persisted to disk by default in v1.

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — product boundaries, dependency flow, subsystem descriptions
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — benchmark subsystem, experiment lanes, Dynamo integration
- [docs/PROFILING.md](docs/PROFILING.md) — runtime profiling surface
- [docs/BENCHMARK-PLAN.md](docs/BENCHMARK-PLAN.md) — benchmark product direction
- [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md) — deployment patterns and MCP workflows
- [docs/GPU-REFERENCE.md](docs/GPU-REFERENCE.md) — NVIDIA Hopper and Blackwell GPU specs
- [docs/GPU-MARKET-MAP.md](docs/GPU-MARKET-MAP.md) — GPU market data, pricing, and fleet estimates
- [docs/INFERENCE-SERVING-REFERENCE.md](docs/INFERENCE-SERVING-REFERENCE.md) — inference serving ecosystem reference
- [CHANGELOG.md](CHANGELOG.md) — version history
- [CONTRIBUTING.md](CONTRIBUTING.md) — development setup and contribution guide
- [SECURITY.md](SECURITY.md) — security policy and threat model
- [VALIDATION.md](VALIDATION.md) — validation reports and test expectations

## License

MIT
