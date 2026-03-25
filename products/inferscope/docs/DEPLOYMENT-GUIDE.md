# InferScope Deployment Guide

This guide describes the current deployment shape of InferScope as it exists in the repository today.

## What InferScope is

InferScope is a self-contained inference optimization product with two operator-facing surfaces:

- the `inferscope` CLI
- the `inferscope serve` MCP server

Both surfaces expose the same core capabilities:

- serving recommendation and engine selection
- config validation and capacity planning
- KV cache and disaggregation guidance
- live diagnostics against real endpoints
- packaged evaluation workloads and benchmark tooling

## Recommended adoption model

### Open-source users

Use InferScope as a local CLI or MCP server that helps choose flags, audit running deployments, and run benchmark workloads against a real endpoint.

Typical flow:

```bash
inferscope recommend Llama-3-70B h200 --workload chat
inferscope validate Llama-3-70B h200 --tp 2 --quantization fp8
inferscope serve
```

### Enterprise users

Treat InferScope as a read-only planning and diagnostics layer in front of your serving fleet.

Typical flow:

1. Use recommendation commands during capacity or rollout planning.
2. Use live diagnostics and audit tools against production-like endpoints.
3. Use packaged benchmark workloads to validate changes before broad rollout.
4. Materialize benchmark stacks when evaluating disaggregation or cache-aware topologies.

## Engine status

| Engine | Status | Notes |
| --- | --- | --- |
| vLLM | Production-oriented in repo | Strongest general-purpose path today |
| SGLang | Production-oriented in repo | Best fit for coding and prefix-heavy workloads |
| ATOM | Production-oriented in repo | AMD-focused MLA/MoE path |
| TRT-LLM | Compiler surface present | Use conservatively until live validation expands |
| Dynamo | Compiler surface present | Use conservatively until live validation expands |

"Compiler surface present" means the code can emit a config or command path, but repo documentation should not oversell live-validated breadth until broader end-to-end validation lands.

## Evaluation subsystem

Benchmark workloads and experiment specs are packaged with the product. Use built-in names rather than repo-relative paths:

```bash
inferscope benchmark-workloads
inferscope benchmark reasoning-chatbot http://localhost:8000
inferscope benchmark-stack-plan vllm-single-endpoint-baseline h100
```

Artifacts default to `~/.inferscope/benchmarks/` so evaluation output stays outside the repo.

## Profiling and kernel work

InferScope now exposes a dedicated profiling boundary in `src/inferscope/profiling/`. Today this is advisory only: the recommender records whether `nsys` or `rocprofv3` is the appropriate next step, but profiler execution and custom kernel work remain future additions.

## Security posture

- CLI commands may allow private endpoints for local operator workflows.
- MCP tools block private addresses by default where network exposure matters.
- Benchmark artifacts and stack bundles are confined to the configured benchmark output directory.
- Authentication helpers normalize bearer, API-key, X-API-Key, and raw header usage for inference and metrics endpoints.

## Validation expectations

See [VALIDATION.md](../VALIDATION.md) for the current validation contract. In short:

- lint, type-check, and security scans are required
- packaged benchmark resources are smoke-tested from a built wheel
- pytest and live integration steps are conditional on the repo actually containing those suites and the required secrets
