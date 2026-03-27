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
- **Dynamo disaggregated serving** — Dynamo 1.0 orchestrated prefill/decode with NIXL KV transfer and KV-aware routing
- **KV cache compression** — FP8 KV quantization, prefix caching hit rates, chunked prefill tuning

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

### NVIDIA ISA reference (Hopper / Blackwell)

| GPU | ISA | Architecture | Key Formats |
|-----|-----|-------------|-------------|
| H100 PCIe | `sm_90` | Hopper | FP8 (E4M3/E5M2), FP16, BF16 |
| H100 SXM / H200 / GH200 | `sm_90a` | Hopper | FP8 (E4M3/E5M2), FP16, BF16, DPX |
| B100 / B200 / GB200 | `sm_100` | Blackwell | FP8, MXFP6, MXFP4, NVFP4, FP16, BF16 |
| B300 / GB300 | `sm_103` | Blackwell Ultra | FP8, MXFP6, MXFP4, NVFP4, FP16, BF16 |

**Cross-architecture compatibility**: PTX compiled for `compute_90a` is **not** forward-compatible with Blackwell. Blackwell adds microscaling formats (MXFP6/MXFP4) and NVFP4 that require native compilation.

### AMD ISA reference (CDNA3 / CDNA4)

| GPU | ISA | Architecture |
|-----|-----|-------------|
| MI300X | `gfx942` | CDNA3 |
| MI355X | `gfx950` | CDNA4 |

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
- TRT-LLM and Dynamo are supported engines with validated gating (Dynamo is RECOMMENDED for multi-node NVIDIA)

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
| `preview_engine` degraded warning | Engine is in preview tier | Switch to a supported engine (`vllm`, `sglang`, `trtllm`, `dynamo`) for production benchmarks |
| Prometheus metrics empty | Endpoint doesn't expose `/metrics` | Verify engine metrics are enabled (vLLM: enabled by default, SGLang: `--enable-metrics`) |
| `tool_parse_success_rate: 0.0` | Model not producing valid JSON tool calls | Check model supports structured output; try a larger model or adjust `--synthetic-output-tokens` |
| Permission denied writing artifacts | `~/.inferscope/benchmarks/` not writable | Set `INFERSCOPE_CACHE_DIR` to a writable path, or `mkdir -p ~/.inferscope/benchmarks` |
| AMD GPU not recognized | GPU name not in registry | Use `mi300x` or `mi355x` as the `--gpu` value; AMD is day-one supported for planning and gating |

## Legacy compatibility

Legacy repo-style references continue to resolve for packaged built-ins, for example:

- `benchmarks/workloads/coding-long-context.yaml`
- `benchmarks/experiment_specs/vllm-single-endpoint-baseline.yaml`

Use the short built-in names for new automation and docs.

## Recommended comparison matrices

### Long-context inference (standard)

1. `vllm-single-endpoint-baseline`
2. `vllm-single-endpoint-offloading-connector`
3. `vllm-disagg-prefill-lmcache-grace`

Progression: GPU-resident baseline → host spill → LMCache/disaggregated overflow.

### Dynamo disaggregated (NIXL KV transfer)

1. `dynamo-disagg-prefill-nixl` — coding workload, Dynamo-orchestrated P/D split with NIXL
2. `dynamo-disagg-prefill-nixl-rag` — long-context RAG (32K–128K), document-switch patterns
3. `dynamo-disagg-prefill-nixl-grace` — Grace Blackwell (GB200/GB300) with HBM + Grace coherent tiers

Each Dynamo experiment spec defines multiple metrics targets:

| Target | Role | Engine | Required | Description |
|--------|------|--------|----------|-------------|
| `primary` | primary | vllm | yes | Decode worker — main latency/throughput metrics |
| `router` | router | dynamo | no | Smart Router — orchestration telemetry |
| `prefill` | prefill | vllm | yes | Prefill worker — KV generation metrics |
| `decode` | decode | vllm | yes | Decode worker — token generation metrics |

The decode worker is the primary metrics endpoint for benchmark results. Router metrics are supplemental orchestration telemetry captured alongside worker snapshots.

### Dynamo end-to-end benchmark workflow

```bash
# 1. Plan the strategy — auto-selects Dynamo lanes for NVIDIA multi-GPU
inferscope benchmark-strategy Qwen3.5-72B h100 \
  --workload coding \
  --num-gpus 4 \
  --has-rdma \
  --multi-node  # optional: promotes Dynamo to RECOMMENDED tier

# 2. Inspect the Dynamo stack plan
inferscope benchmark-stack-plan dynamo-disagg-prefill-nixl h100 --num-gpus 4

# 3. Run the benchmark against a live Dynamo deployment
inferscope benchmark coding-long-context http://localhost:9000 \
  --experiment dynamo-disagg-prefill-nixl \
  --model Qwen3.5-72B \
  --gpu h100 \
  --num-gpus 4

# 4. Compare against a baseline
inferscope benchmark-compare baseline.json dynamo-run.json
```

The stack plan generates launch commands for all Dynamo components:
- **Smart Router** — routes requests based on KV cache state and SLO targets
- **vLLM prefill worker** — handles prompt processing (port 7100)
- **vLLM decode worker** — handles token generation (port 7200)
- **NIXL env wiring** — configures zero-copy RDMA KV transfer between workers

### Multi-node vs multi-GPU semantics

InferScope distinguishes between multi-GPU (single host, multiple GPUs) and multi-node (multiple hosts):

- **Multi-GPU, single host** (e.g., 8×H100 SXM on one node): Dynamo tier = `supported`
- **Multi-node** (e.g., 2+ hosts with RDMA interconnect): Dynamo tier = `recommended`

The `--multi-node` flag is explicit — it is **not** inferred from GPU count or topology mode. This prevents false promotions on single-host multi-GPU systems where Dynamo orchestration overhead may not be justified.

## Artifact location

InferScope benchmark artifacts default to:

```text
~/.inferscope/benchmarks/
```

This keeps runtime output out of the source tree and makes the MCP behavior match the CLI behavior.
