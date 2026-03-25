# InferScope Benchmark Plan

InferScope's benchmark plan is simple:

> turn benchmark methodology into an operator workflow that can be consumed locally or through MCP.

## What this product should do

InferScope should be the fastest path from an operational question to an actionable benchmark loop:

1. recommend a serving strategy
2. replay a relevant workload against an endpoint
3. persist an artifact
4. compare artifacts before and after a change
5. expose the same flow to MCP clients

## What it should not do

InferScope should not become a competing benchmark standard or a dashboard clone.

- InferenceX remains the public reference.
- ISB-1 remains the benchmark standard in this repo.
- InferScope remains the operator layer.

## Current bridge workloads

The immediate bridge workloads are:

- `tool-agent`
- `coding-long-context`

These are high-leverage because they reflect the kinds of workloads operators actually need to validate when using an MCP or coding-focused deployment.

## Donor benchmark basis

The local `inferscope-bench/` tree contributes workload and replay ideas. Those ideas should be absorbed into InferScope's packaged benchmark subsystem rather than maintained as a separate public product.

## Near-term priorities

1. keep the packaged workload catalog self-contained
2. support procedural materialization for bridge workloads
3. preserve stable `WorkloadPack` and `BenchmarkArtifact` contracts
4. make CLI and MCP benchmark surfaces symmetric
5. keep benchmark artifacts easy to review and compare
