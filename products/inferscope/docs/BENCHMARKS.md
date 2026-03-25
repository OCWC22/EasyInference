# InferScope Benchmarks

InferScope ships a packaged benchmark subsystem for operators.

This is distinct from ISB-1:

- **ISB-1** owns the benchmark standard and canonical workload families.
- **InferScope** owns practical built-ins, replay workflows, and artifact handling for CLI and MCP use.

## Relationship to InferenceX

As of **March 25, 2026**, InferenceX is the external public reference for broad, continuously updated hardware/framework results across platforms such as **H100, H200, B200, GB200, GB300, and MI355X**.

InferScope uses that ecosystem reality as an input, but its benchmark layer is intentionally different:

- it is designed to drive **operator decisions**
- it can be consumed directly through **CLI and MCP**
- it emphasizes scenario-specific workloads such as **tool-agent**, **coding-long-context**, and **long-context RAG**
- it shares a recommendation and profiling core with the optimizer instead of acting as a standalone public leaderboard

## Operator extension lanes

InferScope extends beyond the public InferenceX-style matrix by shipping operator-focused benchmark lanes for:

- **realistic long-context RAG**
- **cold-session KV overflow**
- **single-endpoint offload studies**
- **LMCache-backed disaggregated serving**
- **Grace-coherent overflow modeling**

## Built-in benchmark assets

InferScope ships packaged resources under:

- `src/inferscope/benchmarks/workloads/`
- `src/inferscope/benchmarks/experiment_specs/`

These built-ins are available from both the CLI and MCP server.

## Core commands

```bash
# discover built-ins
inferscope benchmark-workloads
inferscope benchmark-experiments
inferscope benchmark-matrix --workload-class tool_agent --engine sglang

# inspect a concrete run plan
inferscope benchmark-plan tool-agent http://localhost:8000

# replay a workload against an endpoint
inferscope benchmark coding-long-context http://localhost:8000

# compare two saved artifacts
inferscope benchmark-compare before.json after.json

# materialize a deployment stack bundle
inferscope benchmark-stack-plan vllm-single-endpoint-baseline h100
inferscope benchmark-stack-plan vllm-single-endpoint-offloading-connector h200
inferscope benchmark-stack-plan vllm-disagg-prefill-lmcache-grace gb200 --num-gpus 4
```

The stack-plan path is important: it uses the same recommendation DAG that powers the MCP. That keeps the benchmark launch plan aligned with the engine/profile decision the operator sees elsewhere.

## Matrix catalog

InferScope now ships a structured benchmark matrix catalog over its packaged assets.

The catalog is intended to answer questions like:

- which workloads target **Blackwell Grace** long-context overflow studies?
- which packaged lanes are focused on **tool-calling** and **SGLang**?
- which experiments are **reference** lanes versus **topology probes**?

The matrix is built from metadata carried directly on packaged workloads and experiment specs:

- `benchmark_role`
- `target_gpu_families`
- `target_model_classes`
- `focus_areas`

Available surfaces:

- CLI: `inferscope benchmark-matrix`
- MCP: `tool_get_benchmark_matrix`

Example:

```bash
inferscope benchmark-matrix \
  --gpu-family blackwell_grace \
  --focus-area kv_offload
```

That returns:

- filtered workload descriptors
- filtered experiment descriptors
- suggested workload/experiment pairings

## Procedural built-ins

Some built-ins can be procedurally expanded at runtime.

Current support:

- `tool-agent`
- `coding-long-context`

Static operator-focused built-ins also ship directly:

- `long-context-kv-offload-rag`

Supported procedural options:

- `--synthetic-requests`
- `--synthetic-input-tokens`
- `--synthetic-output-tokens`
- `--synthetic-seed`
- `--context-file` for `coding-long-context`

Example:

```bash
inferscope benchmark-plan tool-agent http://localhost:8000 \
  --synthetic-requests 8 \
  --synthetic-input-tokens 4096 \
  --synthetic-output-tokens 512
```

```bash
inferscope benchmark coding-long-context http://localhost:8000 \
  --synthetic-requests 4 \
  --synthetic-input-tokens 32768 \
  --synthetic-output-tokens 768 \
  --context-file ./repo_context.txt
```

These commands still resolve to a standard `WorkloadPack` and write a normal `BenchmarkArtifact`.

## Relationship to ISB-1

InferScope built-ins should map back to the stable ISB-1 families rather than inventing a second benchmark taxonomy.

Current mapping:

- `tool-agent` → ISB-1 `agent`
- `coding-long-context` → ISB-1 `coding`
- `long-context-kv-offload-rag` → ISB-1 `rag`

This lets the MCP surface move quickly without destabilizing the benchmark standard.

## Legacy compatibility

Legacy repo-style references continue to resolve for packaged built-ins, for example:

- `benchmarks/workloads/coding-long-context.yaml`
- `benchmarks/experiment_specs/vllm-single-endpoint-baseline.yaml`

Use the short built-in names for new automation and docs.

## Recommended comparison matrix

For long-context inference work, use these experiments together:

1. `vllm-single-endpoint-baseline`
2. `vllm-single-endpoint-offloading-connector`
3. `vllm-disagg-prefill-lmcache-grace`

This gives a practical progression from GPU-resident baseline → host spill → LMCache/disaggregated overflow.

## Artifact location

InferScope benchmark artifacts default to:

```text
~/.inferscope/benchmarks/
```

This keeps runtime output out of the source tree and makes the MCP behavior match the CLI behavior.
