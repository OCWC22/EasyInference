# Changelog

All notable changes to InferScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Validation: [03-runtime-profiling-v1](validations/03-2026-03-25-runtime-profiling-v1.md), [04-hopper-blackwell-hardening](validations/04-2026-03-25-hopper-blackwell-hardening.md)

### Added
- Shared Hopper/Blackwell platform policy used by the recommendation DAG, validators, and compilers
- Explicit support-tier metadata for engine recommendations and compiled engine configs
- New NVIDIA regression coverage for H100, H200, B200, and GB200 recommendation paths
- New benchmark launcher regression coverage to ensure benchmark stack plans inherit the same H200/Hopper policy as the MCP
- Structured benchmark matrix catalog across packaged workloads and experiment specs, exposed through both CLI and MCP
- Benchmark strategy layer that maps model + GPU + workload intent to the right packaged benchmark suite and optional live profiling bridge
- Day-one AMD MI300X (gfx942) and MI355X (gfx950) support for planning, benchmark gating, and support assessment
- AGENTS.md at monorepo root for coding agent onboarding
- Comprehensive SDLC documentation: prerequisites, configuration, test conventions, troubleshooting, and dependency flow diagrams
- New long-context benchmark workload and experiment lanes for:
  - single-endpoint `OffloadingConnector`
  - disaggregated `LMCache` with Grace-aware overflow modeling
  - single-endpoint long-context RAG baseline
  - disaggregated LMCache-backed long-context RAG lane for non-Grace systems

### Changed
- Engine ranking now derives its top pick from the full recommendation DAG instead of a separate heuristic
- DeepSeek Hopper defaults now respect model hints only when they remain memory-valid (`H100 -> AWQ fallback / TP=8`, `H200 -> FP8 / TP=8`)
- Blackwell FP4 recommendations now flow through the main optimizer path
- Grace coherent overflow is surfaced as an advisory memory tier instead of being conflated with plain HBM fit
- Benchmark metadata now has an explicit `grace_coherent` cache tier for realistic long-context operator studies
- Packaged benchmark workloads and experiments now carry explicit role/GPU/model/focus metadata for matrix filtering and catalog discovery

### Fixed
- vLLM compiler no longer infers `GB200` from `192 GB` memory size
- TRT-LLM compiler now uses `batched_token_budget` correctly
- Benchmark launcher workload mapping now recognizes `long_context_rag`
- Benchmark catalog loading now accepts the existing `nixl` experiment lane and resolves experiment workload classes correctly

## [0.1.0] - 2026-03-23

### Added
- Initial release with Phase 1 functionality
- **15 MCP tools** across 5 groups: hardware, model intel, recommendations, KV cache, live diagnostics
- **16 CLI commands**: profile, validate, recommend, gpu, compare, capacity, engine, parallelism, kv-budget, kv-strategy, disagg, quantization, check, memory, cache, serve
- **GPU knowledge base**: 9 variants across 5 architectures (Ampere, Hopper, Blackwell, CDNA3, CDNA4)
- **Model registry**: 12+ models across 5 classes (Dense-GQA, Qwen3.5-Hybrid, Frontier-MLA-MoE, Compact-Agentic-MoE, Classical-MoE)
- **3 engine compilers**: vLLM, SGLang, ATOM (TRT-LLM and Dynamo as stubs)
- Normalized ServingProfile → ConfigCompiler → EngineConfig pipeline
- Memory planner with per-layer KV cache math
- Pre-flight config validation (TP divisibility, memory fit, format compatibility)
- KV cache tiering strategy recommendations (GPU/CPU/SSD)
- Prefill/decode disaggregation decision tool
- Quantization comparison tool with GPU format awareness
- Prometheus metric scraping for vLLM/SGLang/ATOM with auto-detection
- Live diagnostics: check, memory pressure, cache effectiveness
- Input validation and SSRF protection for HTTP endpoints
- 129+ unit tests across 9 test files

### Security
- SSRF protection: private IP blocking, URL scheme validation
- MCP tools block private IPs by default; CLI allows private for local operator use
- Localhost-only binding for HTTP transport by default
- Input validators wired into tool entrypoints (model names, GPU names, numeric bounds)
