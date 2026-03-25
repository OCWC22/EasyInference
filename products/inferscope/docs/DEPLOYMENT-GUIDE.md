# InferScope Deployment Guide

InferScope is meant to sit close to the operator.

Its primary deployment value is not model hosting by itself; it is the ability to profile, validate, tune, and benchmark serving changes through a CLI or MCP.

## Recommended use

### Open-source users

Use InferScope locally to:

- select an engine and serving profile
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

## Runtime storage

Artifacts default to `~/.inferscope/benchmarks/`. Keep this path writable and treat it as operational evidence, not disposable scratch space.

Runtime profiles are returned directly to the CLI or MCP caller in v1. They are not written to disk by default.

## Profiling boundary

`src/inferscope/profiling/` is the isolated seam for runtime profiling today and profiler/kernel work later.

- v1 ships Prometheus-based runtime profiling
- future work can add trace execution helpers and kernel-facing integrations there
- benchmark orchestration should keep consuming shared telemetry models rather than reimplementing profiling logic
