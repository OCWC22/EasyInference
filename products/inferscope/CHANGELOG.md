# Changelog

All notable changes to InferScope will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
