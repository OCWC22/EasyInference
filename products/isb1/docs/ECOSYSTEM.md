# ISB-1 Ecosystem Positioning

This document explains how ISB-1 fits beside InferenceX, InferScope, and the local donor benchmark work.

## The ecosystem roles

### InferenceX

InferenceX is the external, continuously updated public reference for inference serving performance across frameworks and hardware.

It is useful when the question is:

- what does the current market-wide Pareto frontier look like?
- how do major engines compare on shared hardware?
- what is the latest public reference result for a model / framework / accelerator combination?

### ISB-1

ISB-1 is the reproducible benchmark standard inside EasyInference.

It is useful when the question is:

- what exact workload did we replay?
- can we reproduce and defend this result later?
- can we compare baseline, optimized, and operator-submitted configurations under one stable methodology?
- can we publish or review benchmark claims with trace-level evidence?

### InferScope

InferScope is the operator-facing CLI and MCP product.

It is useful when the question is:

- what engine and settings should I use?
- how do I validate a deployment change before rollout?
- how do I expose benchmark replay and artifact comparison through an MCP?
- how do I bridge benchmark methodology into day-to-day tuning work?

## Why EasyInference is not cloning InferenceX

The repository should not become a second dashboard or a loose fork of InferenceX.

Instead:

- let InferenceX remain the public benchmark reference
- let ISB-1 provide a stricter reproducibility and publication surface
- let InferScope provide the high-leverage operator product

That division keeps scope clear and avoids splitting effort across two nearly identical benchmark brands.

## What the local donor benchmark contributes

The local `inferscope-bench/` tree contributes ideas, not a public product contract.

Important donor concepts absorbed into EasyInference:

- MCP or tool-calling request shapes
- long-context coding request shapes
- OpenAI-compatible replay patterns for endpoint benchmarking

These concepts should be translated into the public EasyInference products, not exposed as a third first-class product.

## KV cache benchmarking extension

InferScope extends the ISB-1 workload families with KV-cache-aware benchmark lanes that no public benchmark covers today:

- **KV offloading** — cold-session spill to host DRAM or SSD
- **Disaggregated KV** — prefill/decode split with LMCache or SGLang HiCache
- **Disaggregated serving (Dynamo + NIXL)** — orchestrated prefill/decode with KV-aware routing
- **KV compression** — lz4, fp8, kvtc, turboquant, mxfp4, cachegen overlays
- **Grace-coherent overflow** — 3-tier HBM → Grace LPDDR5X → remote cache

These extensions sit on top of the canonical families without fragmenting the benchmark standard. A Dynamo-orchestrated RAG experiment still belongs to the `rag` family; the cache topology is orthogonal metadata.

## Canonical family mapping

Specific scenarios should map back to stable ISB-1 families.

- `tool-agent` → `agent`
- `coding-long-context` → `coding`
- `long-context-kv-offload-rag` → `rag`
- chat-oriented packaged workloads → `chat`
- retrieval-heavy packaged workloads → `rag`

This keeps the benchmark standard stable while still letting InferScope ship richer built-ins.
