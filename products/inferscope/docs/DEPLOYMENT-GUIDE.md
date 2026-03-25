# InferScope Deployment Guide

InferScope is meant to sit close to the operator.

Its primary deployment value is not model hosting by itself; it is the ability to plan, validate, and benchmark serving changes through a CLI or MCP.

## Recommended use

### Open-source users

Use InferScope locally to:

- select an engine and serving profile
- validate a config before launch
- benchmark a local or remote OpenAI-compatible endpoint
- compare saved artifacts after a tuning change

### Enterprise users

Use InferScope as a read-only planning and validation layer in front of your serving fleet.

Typical flow:

1. generate a recommendation for the target hardware and workload
2. apply the serving change in a staging or canary environment
3. run a benchmark workload through InferScope
4. compare the resulting artifact against a baseline
5. promote only when the artifact review is acceptable

## MCP-first benchmark workflow

This is the highest-leverage deployment pattern in the repo.

The same packaged benchmark subsystem is available through:

- `inferscope benchmark-*` commands
- `inferscope serve` MCP tools

That means a human operator and an MCP client can resolve the same workload, run the same replay, and compare the same artifact model.

## Procedural bridge workloads

Two built-ins matter most today:

- `tool-agent` for MCP and tool-calling flows
- `coding-long-context` for repository-scale coding and review flows

Use them when validating coding agents, MCP servers, or long-context tuning changes.

## Runtime storage

Artifacts default to `~/.inferscope/benchmarks/`. Keep this path writable and treat it as operational evidence, not disposable scratch space.

## Profiling boundary

`src/inferscope/profiling/` is the future seam for profiler and kernel work. Today it is advisory only. Keep profiler-specific or kernel-specific additions isolated there rather than mixing them into benchmark orchestration.
