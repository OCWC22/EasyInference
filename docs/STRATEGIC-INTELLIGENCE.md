# InferScope Strategic Intelligence: Infrastructure, Benchmarking, and Competitive Landscape

> **Audience**: Internal leadership, GPU/cloud/runtime partners, enterprise design partners.
> **Last updated**: March 2026
> **Source of truth**: This document is evidence-backed. Claims are grounded in public sources, not aspirational.

---

## 1. Infrastructure Partner Profiles

### Baseten: the inference-first neocloud

Baseten (commonly misspelled as "Base10") offers **H100** ($6.50/hr) and **B200** ($9.98/hr) GPUs on per-minute billing, plus serverless per-token pricing for open-source models. H200 and B300 are not currently listed. Valued at **$2.15B** after a $150M Series D (September 2025), with NVIDIA investing $150M in a separate $300M round in January 2026, Baseten is firmly positioned as the "Stripe of AI inference" — purpose-built for mission-critical production workloads, not general GPU compute.

Their inference stack selects the optimal framework per workload: TensorRT-LLM, SGLang, or vLLM, with custom Triton kernels and Eagle speculative decoding on top. They publish extensive benchmarks — their BEI embedding runtime achieves **2x higher throughput than vLLM/TEI**, and they claim to win 95% of performance bakeoffs against competitors by 40-50%. Key customers include Cursor, Notion, Writer, Bland AI (sub-400ms voice latency), and Sourcegraph. Baseten holds the top GPU performance spot on OpenRouter's leaderboard. They have **no existing benchmark tool partnership**, making them a prime candidate for third-party performance validation that reinforces their competitive claims.

### Modal: serverless GPU compute with built-in benchmarks

Modal offers the broadest GPU lineup among the three: **B200** ($6.25/hr), **H200** ($4.54/hr), **H100** ($3.95/hr), L40S, A100s, and smaller GPUs — all on pure **per-second serverless billing** with scale-to-zero. B300 is not yet available. Valued at approximately **$2.5B** (February 2026), Modal differentiates through developer experience: a Python-native SDK with sub-second container spin-up and zero infrastructure management.

Notably, Modal has already published **over a thousand LLM engine benchmarks** comparing vLLM and SGLang across GPU types, including a finding that B200 delivers **1.7x higher QPS** than H200 for DeepSeek V3 at median 1-second TTFT. They recommend SGLang for latency-sensitive workloads and vLLM for throughput-first batch workloads. Their startup credits program offers **up to $25K** in free compute. Despite building internal benchmarking capabilities, Modal has **no external benchmarking tool partnership** — an opportunity for InferScope to provide standardized, framework-neutral comparisons on their platform.

### Fireworks AI: proprietary engine, biggest revenue, most to gain from independent validation

Fireworks runs **H100, B200, and AMD MI300** GPUs with a fully **proprietary inference engine** — not vLLM or TensorRT-LLM. Their FireAttention kernel suite (V1-V3) claims **12x faster inference than vLLM**, and their 3D FireOptimizer automates multi-dimensional tradeoffs across speed, cost, quality, hardware type, and parallelism strategy. Processing **13+ trillion tokens/day** at 99.99% uptime, Fireworks serves 10,000+ customers including Samsung, Uber, DoorDash, Cursor, and Sourcegraph.

At **~$280M ARR** and a $4B valuation (October 2025 Series C), Fireworks is the revenue leader among the three. They already reference **Artificial Analysis** rankings as third-party validation, suggesting strong receptivity to independent benchmarking. Their proprietary engine means InferScope would need to benchmark via their OpenAI-compatible API rather than framework-level instrumentation. Key partnership angle: Fireworks' claims of 12x improvement over vLLM need independent verification, and their AMD MI300 deployment is an underexplored benchmarking surface.

**Critical insight across all three partners:** None have existing benchmark partnerships. All publish their own performance data (inherently conflicted). All would benefit from credible independent validation — Baseten to prove their framework selection advantage, Modal to standardize their GPU comparisons, and Fireworks to substantiate their proprietary engine claims against open-source alternatives.

---

## 2. Enterprise Inference Bottlenecks

### KV cache has become the binding constraint

The KV cache — not model weights — is now **the single largest memory consumer** in production inference. For Llama-3 8B with just 40 concurrent 8K-token requests, the KV cache consumes ~40GB versus ~16GB for model weights, pushing total usage to 60GB+ on an 80GB A100. Traditional allocation wastes **60-80%** of KV cache memory through fragmentation; PagedAttention reduced this to under 5%, enabling 2-4x throughput gains.

Per-token KV cache costs are substantial: a 70B model requires **~328KB per token**. At 128K context, that's 42GB per single user session. NVIDIA's NVFP4 KV cache quantization delivers up to **3x better TTFT** and 20% higher cache hit rates compared to FP8, while LMCache achieves **3-10x TTFT reduction** through multi-tier caching (GPU → CPU → disk → S3). VAST Data demonstrated a reduction in TTFT from **11+ seconds to 1.5 seconds** at 128K context using vLLM + LMCache.

### Models in production: closed-source dominates, open-source is the benchmarking surface

The Menlo Ventures mid-2025 survey of 150+ technical leaders shows **Anthropic at 32%** of enterprise production usage, OpenAI at 25% (down from 50% in 2023), and Google at 20%. Open-source models hold only **11-13%** of enterprise production — down from 19% a year prior — partly because Llama 4's April launch "underwhelmed in real-world settings."

However, the models enterprises most need to benchmark for self-hosted deployment are: **Qwen 3.5** (397B MoE, adopted by 90,000+ enterprises and now the most-used base model for fine-tuning), **DeepSeek V3/R1** (671B MoE, input costs as low as $0.07/M tokens with caching), and **Llama 4 Maverick** (400B MoE, natively multimodal). The 37% of enterprises using 5+ models simultaneously creates urgent need for multi-model benchmarking — a gap no current tool addresses.

### Enterprise SLOs are converging on specific targets

Across multiple sources including NVIDIA Dynamo configuration files, MLPerf scenarios, and the DistServe paper, enterprise SLO targets are crystallizing:

- **TTFT**: <200ms for interactive/code, <500ms for standard chat, seconds acceptable for batch
- **TPOT**: <20-30ms for interactive (matching ~1,600 words per minute reading speed), <50ms standard
- **Goodput**: The maximum request rate where ≥P90 of requests meet both TTFT and TPOT thresholds simultaneously
- **P99 tail latency**: Tracked for SLA guarantees; enterprise contracts typically guarantee fast responses for 99% of requests

The DistServe paper from Hao AI Lab (OSDI'24) formalized the critical insight: **"High throughput ≠ High goodput."** A system processing 10 req/s total may only deliver 3 req/s goodput under realistic SLO constraints because aggressive batching inflates per-request latency. This makes goodput-centric benchmarking — where InferScope can lead — more valuable than raw throughput measurement.

---

## 3. Disaggregated Serving: From Research to Production Standard

The Hao AI Lab assessed in November 2025 that **"almost every production-grade LLM serving framework now runs on disaggregation"** — separating prefill and decode phases onto different GPUs for independent optimization. This shift fundamentally changes what benchmarking tools must measure.

**NVIDIA Dynamo 1.0**, announced as production-ready at GTC 2026 (March 16, 2026), is the most comprehensive implementation. It supports disaggregated prefill/decode across SGLang, TensorRT-LLM, and vLLM backends, with KV-aware smart routing, multi-tier KV cache management (GPU → CPU → disk → S3 via NIXL), and dynamic GPU scheduling. Dynamo claims up to **30x token throughput** for DeepSeek-R1 on GB200 NVL72. It is deployed in production by AWS, Azure, Google Cloud, Oracle, plus Baseten, Fireworks, Cursor, Perplexity, and dozens more.

**SGLang v0.5.9** implements native disaggregated serving deployed across **400,000+ GPUs** generating trillions of tokens daily. On 96 H100 GPUs running DeepSeek-R1, SGLang achieves 52.3K input tokens/sec and 22.3K output tokens/sec per node — matching DeepSeek's official numbers. SGLang is expanding to EPD (Encode-Prefill-Decode) disaggregation for multimodal workloads.

**vLLM v0.18.0** labels disaggregated serving as "experimental" but Meta has deployed it at scale internally, achieving **40-50% prefix cache hit rates** with sticky routing and 90% HBM utilization. Meta's implementation outperforms their internal inference stack. The **llm-d** project (Red Hat, Google, IBM, NVIDIA, CoreWeave) provides Kubernetes-native orchestration around vLLM with cache-aware routing, reaching **~3.1K tokens/sec per B200 decode GPU**.

For InferScope, the implication is clear: **any benchmarking tool that only measures single-node, colocated serving is already obsolete.** Benchmarks must capture prefill-decode separation, KV transfer overhead, and the goodput impact of disaggregation topology choices.

---

## 4. The Benchmarking Landscape: Exploitable Gaps

### InferenceX dominates but has known limitations

InferenceX (formerly InferenceMAX, SemiAnalysis) is the most widely cited independent inference benchmark, running **nightly CI/CD benchmarks** across NVIDIA H100/H200/B200/GB200/GB300 and AMD MI300X/MI325X/MI355X using vLLM, SGLang, and TensorRT-LLM. Both NVIDIA and AMD publish official responses to InferenceX results. Its throughput-versus-interactivity Pareto frontier methodology captures a real tradeoff, and its TCO modeling uses SemiAnalysis's proprietary AI TCO Model surveying 70+ GPU clouds.

However, InferenceX has acknowledged gaps that InferScope can exploit:

- **No real-world dataset support yet** — uses synthetic fixed ISL/OSL pairs, not production traces
- **No multi-model or multi-tenant benchmarks** — tests one model at a time
- **No agentic/tool-use benchmarks** — planned but not implemented
- **Proprietary TCO model** — not reproducible by the community
- **Software snapshot problem** — results reflect a point-in-time optimization state (AMD's forked vLLM images have lagged official releases, causing controversy)

### Existing tools leave critical gaps unfilled

The tool landscape includes vLLM's benchmark_serving.py (de facto academic standard), SGLang's bench_serving.py (most feature-rich with LoRA and PD-disagg support), NVIDIA's GenAI-Perf/AIPerf (best UX and goodput support, but being deprecated/transitioned), Anyscale's LLMPerf (provider benchmarking, less maintained), and Red Hat's GuideLLM (enterprise-oriented).

No single tool provides all of: production trace replay with burstiness, standardized goodput measurement, multi-model benchmarking, GPU telemetry, and open TCO modeling. **GenAI-Perf's deprecation in favor of AIPerf** creates a specific window — the incumbent NVIDIA-endorsed tool is in transition, and enterprises need stable alternatives.

### Production trace datasets exist but need synthesis

The benchmarking community lacks a dataset combining real arrival patterns with actual prompt content. **BurstGPT** (10.31M traces, 213 days of Azure OpenAI data) provides realistic arrival patterns and burstiness but contains no prompt content. **ShareGPT** provides conversation content but no timestamps. **Azure LLM Inference traces** (2023-2025) offer timestamps and token counts but no content and limited duration. **LMSYS-Chat-1M** has content from 25 LLMs but no arrival patterns.

InferScope's opportunity: **merge BurstGPT arrival patterns with ShareGPT/LMSYS content distributions** to create the first realistic end-to-end benchmark workload — addressing the most frequently cited gap in the field.

---

## 5. GPU Access: Achievable Through Stacked Credit Programs

### Credit programs can fund initial benchmarking

The most efficient path to GPU access for an open-source benchmarking tool:

1. **NVIDIA Inception** (free, no equity): Preferred GPU pricing plus up to **$100K AWS credits** via AWS Activate partnership, and up to **$150K Nebius credits**
2. **Google Cloud AI track**: Up to **$350K** over 2 years for AI-focused pre-Series B startups
3. **AWS Activate**: Up to **$300K** for GenAI-focused applicants (via VC referral)
4. **Prime Intellect Fast Compute Grants**: **$500-$100K** specifically for open-source AI projects
5. **a16z Open Source AI Grant**: Direct grant funding (SGLang received this in June 2025)
6. **Modal**: $25K startup credits; $30/month free for any developer

InferenceX obtained its GPU access through direct hardware sponsorship from NVIDIA (GB200 NVL72, B200 GPUs) and AMD (MI355X), plus compute from Crusoe, CoreWeave, Nebius, TensorWave, Oracle, and Together AI. SGLang secured the a16z Open Source AI Grant and has partnerships with AMD, NVIDIA, and multiple neoclouds.

### Benchmarking costs are manageable

A comprehensive benchmark cycle across H100, H200, and B200 — testing 4-5 models, multiple concurrency levels, 3 ISL/OSL configurations, with 3-10 runs for statistical significance — requires approximately **300-675 GPU-hours** total. At current spot pricing (H100: **$1.49-2.17/hr** on Vast.ai/RunPod, H200: **$3.35-3.72/hr** on RunPod/GCP spot, B200: **$2.25-3.99/hr** on Spheron/DataCrunch), the total cost per benchmark cycle ranges from **$2,000-$11,000**. This is well within reach of a single credit program allocation.

---

## 6. LMCache: Critical Ecosystem Piece

**LMCache** — created at the University of Chicago and now a **PyTorch Foundation ecosystem project** — has emerged as the de facto KV cache management layer for enterprise inference. It provides multi-tier caching (GPU → CPU → disk → S3), cross-instance KV sharing, and disaggregated prefill support across vLLM, SGLang, and NVIDIA Dynamo.

Performance claims include **3-10x TTFT reduction** and **up to 15x throughput improvement**. P2P multi-node CPU memory sharing moved from experimental to production in January 2026. The 2026 Q1 roadmap includes TRTLLM and Modular integration, AMD GPU acceleration, io_uring optimization, and a standalone "LMCache Operator" container separating KV cache management from inference engine processes.

For InferScope, LMCache is both a benchmarking target (how much does LMCache improve performance under various workloads?) and a potential integration partner (LMCache's KV Events protocol could feed benchmark-level observability). Its adoption by GMI Cloud, Google Cloud, CoreWeave, and Tencent signals production maturity.

---

## 7. Strategic Positioning Summary

The inference benchmarking landscape in early 2026 has a clear structural gap. InferenceX owns the hardware-centric, vendor-relationship-driven benchmarking niche but lacks production realism and open methodology. NVIDIA's GenAI-Perf → AIPerf transition creates instability in the enterprise toolchain. No tool combines goodput-first measurement, real trace replay, multi-model benchmarking, and disaggregated serving awareness.

### Five strategic moves

1. **Lead with goodput, not throughput.** The DistServe-originated goodput metric is now the enterprise standard, but only GenAI-Perf/AIPerf and vLLM's benchmark_serving.py partially implement it. InferScope should make configurable, SLO-driven goodput the core differentiator.

2. **Build the missing trace-replay capability.** Combining BurstGPT arrival patterns with ShareGPT/LMSYS content creates the first production-realistic benchmark workload — the single most cited gap across all existing tools.

3. **Target Baseten, Modal, and Fireworks sequentially.** All three lack benchmark partnerships. Baseten's framework-selection approach and published bakeoff claims make them the easiest initial partner (they want third-party validation). Modal's existing benchmark infrastructure makes them a natural platform partner. Fireworks' proprietary engine and highest revenue create the strongest incentive for independent benchmarking.

4. **Stack GPU credits aggressively.** NVIDIA Inception + AWS Activate + Google Cloud AI track can yield **$500K+** in credits at zero equity cost, more than sufficient for continuous benchmarking. Neocloud partnerships (following InferenceX's model) can supplement with dedicated hardware access.

5. **Benchmark disaggregated serving from day one.** With Dynamo 1.0 production-ready, SGLang on 400K+ GPUs, and Meta running vLLM disagg at scale, any new benchmarking tool that only measures colocated serving is launching into obsolescence. PD-disaggregated topologies, KV transfer overhead, and cache-aware routing effectiveness should be first-class metrics.
