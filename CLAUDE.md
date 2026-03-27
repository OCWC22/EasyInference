# CLAUDE.md — Strategic Context for Claude Code

This file gives Claude Code the decision-making context it needs to work on the EasyInference monorepo. AGENTS.md tells you where to put code. This file tells you what to build and why.

## Identity

EasyInference is a two-product monorepo. The product that matters right now is **InferScope** — an open-source inference benchmarking and optimization tool. The other product (ISB-1) is a benchmark standard that InferScope consumes.

InferScope is built by a solo founder (William Chen) who is a GPU kernel optimization engineer. The tool is pre-revenue, pre-launch, and being prepared for partnerships with infrastructure providers and enterprise teams.

## Strategic Position (March 2026)

InferScope is complementary to InferenceX (SemiAnalysis), not competitive:

- **InferenceX** = public leaderboard with synthetic workloads, proprietary TCO models, nightly CI on 1000+ GPUs
- **InferScope** = operator tool with production trace replay, goodput measurement, KV cache telemetry, and the "why" layer

The gap InferScope fills: no existing tool combines (1) real trace replay with production arrival patterns, (2) standardized goodput/SLO measurement, (3) KV cache observability, and (4) multi-engine comparison in a single open-source package.

## Active Partnership: SkyPilot (UC Berkeley)

SkyPilot is actively interested in a benchmark collaboration. This is the most important near-term partnership.

### What SkyPilot enables
- **Multi-cloud GPU orchestration** — launch benchmark runs on the cheapest available H100/H200/B200 across AWS, GCP, Azure, Lambda, RunPod, etc.
- **Spot instance management** — automatic preemption recovery, checkpointing, failover
- **Reproducible environments** — YAML-based cluster definitions with exact package versions
- **Cost optimization** — find cheapest GPU-hours across 12+ cloud providers automatically

### What the collaboration should produce
1. **SkyPilot task YAMLs** for InferScope benchmark runs (launch vLLM/SGLang + run InferScope in one command)
2. **A shared benchmark suite** that SkyPilot uses to validate their multi-cloud performance claims
3. **Published results** comparing GPU providers/generations that benefit both projects
4. **Integration path** where `inferscope benchmark` can optionally use SkyPilot to provision GPUs on-demand

### Architecture decisions driven by SkyPilot
- Benchmark runs must be **stateless and resumable** — SkyPilot spot instances can be preempted
- Artifact persistence must work with **cloud object storage** (S3/GCS), not just local disk
- Environment setup must be **deterministic** — pinned vLLM/SGLang versions in reproducible containers
- Results must be **machine-comparable** — same workload + same model + different hardware = valid comparison

### What NOT to build for SkyPilot
- Do NOT build a SkyPilot SDK integration inside InferScope's core — keep it as external task YAMLs
- Do NOT make SkyPilot a required dependency — InferScope must work standalone against any endpoint
- Do NOT build cloud provider abstractions — SkyPilot already does this

## Honesty Boundaries (Critical)

Claude Code must internalize these. They come from the [AUDIT-BENCHMARK-REALITY.md](products/inferscope/docs/AUDIT-BENCHMARK-REALITY.md) self-audit.

### What is REAL and works today
- HTTP benchmark replay against any OpenAI-compatible endpoint (`benchmarks/runtime.py`, ~920 lines)
- TTFT/TPOT/ITL/goodput measurement from real streaming SSE responses
- Production dataset loading: ShareGPT, any HuggingFace dataset, custom JSONL (`benchmarks/datasets.py`)
- Prometheus metrics capture from vLLM/SGLang endpoints (before/after snapshots)
- CLI and MCP surfaces are end-to-end callable
- Support gating for GPU/model/engine combinations
- Stack plan generation with launch scripts and readiness probes

### What is FAKE or aspirational — never claim these work
- **Template workloads are stubs**: `coding-long-context.yaml` claims 32K tokens but has ~80 real tokens. `long-context-kv-offload-rag.yaml` claims 96K with ~150 tokens. The payloads literally say "Imagine this block contains 96,000 tokens..."
- **Dynamo orchestration**: planning-only, never run on real hardware. DynamoAdapter returns empty for detect/metrics/config
- **LMCache disaggregated prefill**: config generation exists but NOT validated on hardware. Port values may be wrong
- **Continuous metrics**: only before/after Prometheus snapshots, no time-series during load
- **96K RAG benchmark**: the actual HTTP payload is ~150 tokens, not 96K

### Decision rule for Claude Code
When implementing features or writing docs:
- If it touches the benchmark runtime (`runtime.py`, `openai_replay.py`) → this is real, be precise
- If it touches launchers/stack plans → this is planning metadata, clearly label as "generates launch commands, not validated on hardware"
- If it touches workload payloads → distinguish between `hydration:template` (stub) and `hydration:hydrated` (real data)
- If it touches Dynamo → label as experimental/planning-only
- Never write docs claiming InferScope "benchmarks 96K token RAG workloads" — it benchmarks whatever payload you give it, which may be a 150-token stub

## Priority Stack (What to Build Next)

### P0: Make the SkyPilot collaboration concrete
1. SkyPilot task YAML templates for: (a) single-GPU vLLM benchmark, (b) multi-GPU SGLang benchmark, (c) disaggregated prefill/decode benchmark
2. Artifact upload to cloud storage after benchmark completion
3. Pinned environment definitions (vLLM version, CUDA version, model weights)
4. README and example showing: `sky launch inferscope-bench.yaml` → results in S3

### P1: Fix the workload honesty gap
1. Integrate BurstGPT arrival patterns with ShareGPT content for realistic trace replay
2. Add WildChat-4.8M as a dataset source (real conversations with timestamps AND token counts)
3. Make `--dataset` the primary benchmark path, not template workloads
4. Kill or clearly demote the fake 96K/32K template workloads in docs and defaults

### P2: Continuous metrics during benchmark runs
1. Sample Prometheus `/metrics` at configurable intervals (e.g., every 2s) during benchmark execution
2. Capture KV cache utilization, queue depth, and preemption counts as time-series
3. Store in the BenchmarkArtifact alongside before/after snapshots
4. This is the #1 technical differentiator vs InferenceX

### P3: Goodput as the headline metric
1. Make SLO-driven goodput the primary output, not just a field in observed_runtime
2. Support tiered TTFT/TPOT thresholds by prompt length (already partially implemented)
3. Add goodput vs request-rate sweep capability (find the knee of the curve)
4. Align with DistServe's goodput methodology

## Models to Prioritize

For SkyPilot collaboration benchmarks, focus on these (they represent the three main architecture families):

| Model | Architecture | Why |
|-------|-------------|-----|
| Qwen3.5-32B | Dense hybrid | Most-used base for fine-tuning, represents the dense GQA family |
| DeepSeek-R1 | MoE + MLA | Represents frontier reasoning, MLA attention is architecturally distinct |
| Llama-3.3-70B-Instruct | Dense GQA | Universal reference model, everyone has results for comparison |
| Qwen3.5-235B (MoE) | Large MoE | Represents the large MoE serving challenge |

## GPUs to Prioritize for SkyPilot Benchmarks

| GPU | Priority | Why |
|-----|----------|-----|
| H100 SXM | P0 | Universal baseline, most available, most compared |
| H200 SXM | P1 | Memory bandwidth story (4.8 TB/s vs 3.35), shows decode improvement |
| B200 | P1 | Blackwell generation, NVFP4, 192GB HBM3e |
| A100 80GB | P2 | Legacy comparison point, still widely deployed |

## Enterprise Workload Categories (for conversation prep)

When implementing benchmark workloads, these are the categories enterprise teams actually need:

1. **Coding agent** — Cursor/Devin-style: growing repo context, tool calls, 32K-128K windows, prefix cache critical
2. **RAG** — Document chunks per query, variable context, prefix sharing across queries
3. **Agentic** — Multi-step tool-calling, expanding context per turn, session affinity
4. **Multi-turn chat** — Session history accumulation, consumer chatbot patterns

## Key Technical Decisions

### Benchmark runtime is the product
The ~920 lines in `benchmarks/runtime.py` plus `openai_replay.py` are the core product. Everything else (strategy planner, launcher, matrix catalog) is planning metadata. Protect the runtime's correctness above all.

### Goodput > Throughput
InferenceX already owns throughput-vs-latency Pareto curves. InferScope's differentiation is goodput (% of requests meeting SLO) + the "why" layer (what's causing SLO violations). Every benchmark output should lead with goodput.

### Real traces > Synthetic workloads
The `datasets.py` module (ShareGPT, HuggingFace, JSONL loading) is more valuable than the template workload YAMLs. Prioritize making dataset loading robust over expanding synthetic workloads.

### Prometheus telemetry is the moat
No other open-source benchmark tool captures engine-level KV cache metrics alongside request-level latency. This is the "why" layer. Invest in making it continuous (time-series during load) rather than just before/after snapshots.

## Development Commands

```bash
# InferScope
cd products/inferscope
uv sync --dev
uv run pytest tests/ -v --tb=short          # unit tests
uv run ruff check src/ tests/               # lint
uv run ruff format --check src/ tests/      # format
uv run mypy src/inferscope/                 # types
uv run bandit -r src/inferscope/ -c pyproject.toml -ll  # security

# Quick smoke test
uv run inferscope benchmark-workloads
uv run inferscope benchmark-experiments
```

## File Conventions

- Test files: `tests/test_<module>_<area>.py`
- Use `httpx.MockTransport` for HTTP mocking, not `unittest.mock.patch`
- Use `pytest.mark.asyncio` for async tests
- Never import from `inferscope-bench/` (donor, not dependency)
- Workload YAMLs in `src/inferscope/benchmarks/workloads/`
- Experiment specs in `src/inferscope/benchmarks/experiment_specs/`

## Related Documents

- [AGENTS.md](AGENTS.md) — directory routing, dependency direction, dev setup, test conventions
- [products/inferscope/CLAUDE.md](products/inferscope/CLAUDE.md) — low-level implementation guide for InferScope
- [docs/STRATEGIC-INTELLIGENCE.md](docs/STRATEGIC-INTELLIGENCE.md) — infrastructure partner profiles, benchmarking landscape, GPU access strategy
- [products/inferscope/docs/AUDIT-BENCHMARK-REALITY.md](products/inferscope/docs/AUDIT-BENCHMARK-REALITY.md) — detailed honesty audit of what works vs what's aspirational

## Update This File When

- Monorepo product structure changes (new product added, product removed)
- Primary GPU/model targets change (new generation, model decommissioned)
- The runnable vs planning-only boundary changes materially (e.g., Dynamo becomes validated)
- A new partnership becomes active or a partnership changes scope
- Priority stack order changes
