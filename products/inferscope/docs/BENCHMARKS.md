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

## Serving runtime

InferScope now uses a packaged serving runtime instead of a minimal replay loop.

That runtime pulls in the useful mechanics from the local InferenceX-derived donor client and adapts them to InferScope's packaged workloads:

- scheduled arrivals (`immediate`, `poisson`, `gamma`)
- warmup requests
- session-aware replay
- per-request TTFT
- TPOT-style decode latency
- ITL from streamed output events
- request/output throughput
- optional goodput thresholds
- tool-call parse success for MCP-style workloads

The resulting artifact still keeps the existing `BenchmarkArtifact` contract, but richer runtime metrics are written under:

- `run_plan.execution`
- `run_plan.support`
- `run_plan.observed_runtime`

This keeps older artifact readers usable while giving the MCP enough signal to optimize real deployments.

## GPU / model / ISA support gating

Benchmark planning and execution are now support-aware.

The CLI and MCP benchmark tools can validate:

- GPU SKU (NVIDIA: H100, H200, B200, GB200, etc.; AMD: MI300X, MI355X)
- GPU ISA / compute capability (`sm_90a`, `sm_100`, `sm_103` for NVIDIA; `gfx942`, `gfx950` for AMD)
- platform family (`hopper`, `hopper_grace`, `blackwell_grace`, `cdna3`, `cdna4`, etc.)
- model class
- engine support tier
- topology compatibility
- cache / transport compatibility

Support states:

- `supported`
- `degraded`
- `unsupported`
- `unknown`

Examples:

- Grace-coherent lanes reject non-Grace GPUs
- `OffloadingConnector` lanes require single-endpoint vLLM
- `LMCache` lanes require split topology
- `NIXL` lanes degrade when neither RDMA nor a high-speed interconnect is available
- preview engines such as TRT-LLM and Dynamo surface as degraded rather than silently passing

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

# inspect a support-aware plan for a specific GPU / model / engine
inferscope benchmark-plan long-context-kv-offload-rag http://localhost:8000 \
  --experiment vllm-single-endpoint-long-context-rag-baseline \
  --model Qwen3.5-72B \
  --gpu gb200 \
  --num-gpus 4 \
  --engine vllm

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

## Benchmark strategy layer

InferScope now builds directly on top of the packaged benchmark catalog to plan the right operator workflow.

The benchmark-strategy surface answers:

- which workload should be primary for this model + GPU + workload mode?
- which benchmark lanes should be compared first?
- when should the operator run baseline vs offload vs disaggregated studies?
- how should a live runtime profile reorder that suite?

Available surfaces:

- CLI: `inferscope benchmark-strategy`
- MCP: `tool_plan_benchmark_strategy`

Example:

```bash
inferscope benchmark-strategy Qwen3.5-72B gb200 \
  --workload long_context_rag \
  --num-gpus 4 \
  --avg-prompt-tokens 32768 \
  --endpoint http://localhost:8000
```

This bridges three layers:

1. optimizer recommendation
2. packaged benchmark suite selection
3. runtime profiling and tuning preview

For long-context RAG on our benchmark, the intended progression is now:

1. `vllm-single-endpoint-long-context-rag-baseline`
2. `vllm-single-endpoint-offloading-connector`
3. `vllm-disagg-prefill-lmcache-rag`
4. `vllm-disagg-prefill-lmcache-grace` on Grace systems

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

## MCP surfaces

The benchmark MCP is now meant to be directly usable for real benchmark orchestration, not just catalog lookup.

Important surfaces:

- `tool_resolve_benchmark_plan`
- `tool_run_benchmark`
- `tool_plan_benchmark_strategy`
- `tool_generate_benchmark_stack_plan`

These now return or embed:

- support status
- GPU ISA
- engine support tier
- benchmark execution settings
- observed runtime metrics after execution

## Relationship to ISB-1

InferScope built-ins should map back to the stable ISB-1 families rather than inventing a second benchmark taxonomy.

Current mapping:

- `tool-agent` → ISB-1 `agent`
- `coding-long-context` → ISB-1 `coding`
- `long-context-kv-offload-rag` → ISB-1 `rag`

This lets the MCP surface move quickly without destabilizing the benchmark standard.

## Troubleshooting

Common issues when running benchmarks:

| Problem | Cause | Fix |
|---------|-------|-----|
| `Connection refused` on benchmark run | Endpoint not running or wrong URL | Verify the serving endpoint is up: `curl http://localhost:8000/v1/models` |
| `Model not found` in plan resolution | Model name doesn't match registry | Use `inferscope benchmark-plan <workload> <endpoint> --model <name>` with an exact registry name, or check `inferscope recommend --list-models` |
| `unsupported` status in support payload | GPU/topology/engine combination is gated | Check the `issues` array in the support response — each issue has a `code` and `reason` explaining the gate |
| `degraded` status for NIXL transport | No RDMA / high-speed interconnect detected | Expected on commodity networks — benchmark still runs but results may not reflect production performance |
| `preview_engine` degraded warning | TRT-LLM or Dynamo selected | These are preview planning targets — switch to `vllm` or `sglang` for production benchmarks |
| Prometheus metrics empty | Endpoint doesn't expose `/metrics` | Verify engine metrics are enabled (vLLM: enabled by default, SGLang: `--enable-metrics`) |
| `tool_parse_success_rate: 0.0` | Model not producing valid JSON tool calls | Check model supports structured output; try a larger model or adjust `--synthetic-output-tokens` |
| Permission denied writing artifacts | `~/.inferscope/benchmarks/` not writable | Set `INFERSCOPE_CACHE_DIR` to a writable path, or `mkdir -p ~/.inferscope/benchmarks` |
| AMD GPU not recognized | GPU name not in registry | Use `mi300x` or `mi355x` as the `--gpu` value; AMD is day-one supported for planning and gating |

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
