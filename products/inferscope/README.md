# InferScope

**A self-contained MCP and CLI for inference optimization tuning.** InferScope takes a model, GPU, and workload, then returns an engine recommendation, serving profile, launch command, and diagnostic guidance grounded in real hardware constraints.

Within the EasyInference monorepo, the InferScope product lives at `products/inferscope/`.

## Product shape

InferScope is now organized around three explicit layers:

1. **Core optimizer** — `optimization/`, `engines/`, `hardware/`, `models/`, and `tools/`
2. **Evaluation subsystem** — packaged workloads, experiments, replay, artifact comparison, and stack planning under `src/inferscope/benchmarks/`
3. **Future profiling boundary** — `src/inferscope/profiling/`, which currently records advisory `nsys` / `rocprofv3` intent without mixing kernel work into the main product

This keeps the repo usable as a clean open-source package today while preserving a future seam for a separate kernel or profiler repo later.

## What InferScope does

- **Recommend exact serving configs** for vLLM, SGLang, ATOM, TRT-LLM, or Dynamo
- **Validate configs before deploy** for topology, memory, and format fit
- **Plan KV cache and disaggregation strategy** using workload-aware heuristics
- **Audit live endpoints** via Prometheus-backed diagnostics and MCP tools
- **Run packaged evaluation workloads** and compare benchmark artifacts
- **Materialize benchmark stacks** for cache-aware or disaggregated serving experiments

## Quick start

```bash
# Install from PyPI
pip install inferscope

# Or work from source in this monorepo
cd EasyInference/products/inferscope
uv sync --dev

# Core optimization
inferscope recommend DeepSeek-R1 h100 --num-gpus 8 --workload coding
inferscope validate Llama-3-70B h200 --tp 2 --quantization fp8
inferscope kv-strategy Llama-3-70B h100 --workload agent
inferscope audit http://localhost:8000 --gpu-arch sm_90a --model-name Llama-3-70B

# Evaluation subsystem
inferscope benchmark-workloads
inferscope benchmark-experiments
inferscope benchmark coding-long-context http://localhost:8000
inferscope benchmark-plan reasoning-chatbot http://localhost:8000
inferscope benchmark-stack-plan vllm-single-endpoint-baseline h100

# MCP server
inferscope serve
```

## Benchmark assets are packaged

Built-in workloads and experiments now live under `src/inferscope/benchmarks/` and ship inside the package. That means:

- the installed wheel is self-contained
- the MCP server and CLI resolve the same built-ins everywhere
- repo-root benchmark mirrors are no longer the source of truth

Custom YAML files are still supported. Legacy references such as `benchmarks/workloads/coding-long-context.yaml` are accepted for compatibility, but the preferred interface is the built-in name (`coding-long-context`).

See [docs/BENCHMARKS.md](docs/BENCHMARKS.md) for benchmark usage and artifact conventions.

## MCP integration

```bash
# Claude Code
claude mcp add inferscope -- inferscope serve

# Or via uvx
claude mcp add inferscope -- uvx inferscope serve
```

Once connected, your MCP client can call recommendation, diagnostics, and benchmark tools directly.

## Engine coverage

| Engine | Status in repo | Best fit |
| --- | --- | --- |
| vLLM | Mature compiler + diagnostics path | General-purpose serving |
| SGLang | Mature compiler + diagnostics path | Coding, agent, prefix-heavy workloads |
| ATOM | Mature AMD-focused compiler path | Frontier MLA/MoE on AMD |
| TRT-LLM | Compiler surface present, live validation pending | High-throughput NVIDIA deployments |
| Dynamo | Compiler surface present, live validation pending | Multi-node NVIDIA orchestration |

## Repo boundaries

- **Authoritative built-ins:** `src/inferscope/benchmarks/workloads/` and `src/inferscope/benchmarks/experiment_specs/`
- **Benchmark artifacts and stack bundles:** `~/.inferscope/benchmarks/` by default
- **Public surfaces:** `inferscope` CLI and `inferscope serve` MCP server
- **Future profiler/kernel boundary:** `src/inferscope/profiling/`

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — subsystem layout and DAG flow
- [docs/BENCHMARKS.md](docs/BENCHMARKS.md) — packaged benchmark usage and artifact model
- [docs/BENCHMARK-PLAN.md](docs/BENCHMARK-PLAN.md) — evaluation roadmap and measurement philosophy
- [docs/DEPLOYMENT-GUIDE.md](docs/DEPLOYMENT-GUIDE.md) — practical deployment guidance for open-source and enterprise adopters
- [VALIDATION.md](VALIDATION.md) — current validation contract
- [SECURITY.md](SECURITY.md) — security posture and operational constraints

## License

MIT
