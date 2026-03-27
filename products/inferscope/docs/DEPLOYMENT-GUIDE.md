# InferScope Deployment Guide

InferScope is meant to sit close to the operator.

Its primary deployment value is not model hosting by itself; it is the ability to profile, validate, tune, and benchmark serving changes through a CLI or MCP.

## Recommended use

### Open-source users

Use InferScope locally to:

- select an engine and serving profile (vLLM/SGLang on NVIDIA, vLLM on AMD)
- validate a config before launch
- profile a local or remote endpoint through Prometheus metrics
- benchmark a local or remote OpenAI-compatible endpoint
- compare saved artifacts after a tuning change

### Enterprise users

Use InferScope as a read-only planning and validation layer in front of your serving fleet.

Typical flow:

1. generate a recommendation for the target hardware and workload
2. profile the live or staging endpoint
3. review bottlenecks and audit findings
4. preview scheduler/cache tuning changes
5. benchmark before and after the change
6. compare artifacts and promote only when the review is acceptable

## MCP-first workflow

This is the highest-leverage deployment pattern in the repo.

The same packaged profiling and benchmark subsystems are available through:

- `inferscope profile-runtime`, `audit`, `check`, `memory`, `cache`
- `inferscope benchmark-*` commands
- `inferscope serve` MCP tools

That means a human operator and an MCP client can inspect the same live endpoint, resolve the same workload, run the same replay, and compare the same artifact model.

## Procedural bridge workloads

Two built-ins matter most today:

- `tool-agent` for MCP and tool-calling flows
- `coding-long-context` for repository-scale coding and review flows

Use them when validating coding agents, MCP servers, or long-context tuning changes.

## Dynamo / disaggregated deployments

For deployments using NVIDIA Dynamo 1.0 for disaggregated prefill/decode serving:

- Ensure RDMA or NVLink connectivity between prefill and decode GPU pools for NIXL KV transfer
- Use the packaged Dynamo experiment specs (3 available) to benchmark disaggregated throughput and KV transfer latency
- The benchmark strategy layer auto-selects Dynamo lanes for multi-GPU NVIDIA deployments
- Dynamo is RECOMMENDED for multi-node NVIDIA and SUPPORTED for single-node configurations
- The Smart Router provides KV-aware scheduling and SLO-driven autoscaling out of the box
- Compression overlays (lz4, fp8, kvtc, turboquant, mxfp4, cachegen) are available for bandwidth-constrained environments

### Dynamo component architecture

A Dynamo disaggregated deployment consists of:

| Component | Port (default) | Role |
|-----------|---------------|------|
| Smart Router | 9000 (API), 9100 (metrics) | KV-aware request routing, SLO planning |
| vLLM Prefill Worker | 7100 | Prompt processing, KV cache generation |
| vLLM Decode Worker | 7200 | Token generation from cached KV state |
| NIXL | (internal) | Zero-copy RDMA/NVLink KV transfer between workers |

### Single-host vs multi-node

| Scenario | Example | Dynamo Tier | Notes |
|----------|---------|-------------|-------|
| Single-host, 4×H100 | DGX H100 | `supported` | Dynamo works but overhead may not be justified |
| Single-host, 8×H100 SXM | DGX H100 | `supported` | NVSwitch provides fast intra-node KV transfer |
| Multi-node, 2×DGX H100 | 16×H100 across nodes | `recommended` | RDMA enables cross-node NIXL KV transfer |
| Multi-node, GB200 NVL72 | 72×GB200 rack | `recommended` | Production target — 7× perf boost with disaggregation |

Use `--multi-node` explicitly in strategy/recommend commands. GPU count alone does not imply multi-node.

### ISA targets for Dynamo benchmarks

Dynamo benchmarks validate against specific GPU ISAs:

| GPU | ISA | Platform Family | Dynamo Support |
|-----|-----|----------------|----------------|
| H100 PCIe | `sm_90` | hopper | supported |
| H100 SXM | `sm_90a` | hopper | supported |
| H200 | `sm_90a` | hopper | supported |
| GH200 | `sm_90a` | hopper_grace | supported |
| B200 | `sm_100` | blackwell | supported |
| GB200 | `sm_100` | blackwell_grace | supported (recommended multi-node) |
| B300 | `sm_103` | blackwell_ultra | supported |
| GB300 | `sm_103` | blackwell_ultra_grace | supported (recommended multi-node) |

**Important**: PTX compiled for `compute_90a` (Hopper) is **not** compatible with Blackwell (`sm_100`/`sm_103`). Blackwell introduces MXFP6/MXFP4 microscaling formats and NVFP4 that are not available on Hopper.

## Runtime storage

Artifacts default to `~/.inferscope/benchmarks/`. Keep this path writable and treat it as operational evidence, not disposable scratch space.

Runtime profiles are returned directly to the CLI or MCP caller in v1. They are not written to disk by default.

## GPU platform support

- **NVIDIA Hopper/Blackwell** (H100, H200, B200, GB200): primary validated path, full recommendation + profiling + benchmark support
- **AMD MI300X / MI355X**: day-one support for planning, benchmark gating, and support assessment; profiling support follows NVIDIA parity
- GPU telemetry: DCGM (port 9400) for NVIDIA, AMD DME (port 5000) for AMD — both assumed on a trusted network

## Profiling boundary

`src/inferscope/profiling/` is the isolated seam for runtime profiling today and profiler/kernel work later.

- v1 ships Prometheus-based runtime profiling
- future work can add trace execution helpers and kernel-facing integrations there (`nsys` for NVIDIA, `rocprofv3` for AMD)
- benchmark orchestration should keep consuming shared telemetry models rather than reimplementing profiling logic
