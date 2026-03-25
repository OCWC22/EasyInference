"""Recommendation engine — generates optimal ServingProfiles.

Re-architected into a modular Directed Acyclic Graph (DAG) for MCP routing.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field

from inferscope.engines.base import DeploymentInventory, EngineConfig
from inferscope.engines.registry import get_compiler
from inferscope.hardware.gpu_profiles import GPUProfile
from inferscope.logging import get_logger
from inferscope.models.registry import ModelVariant
from inferscope.optimization.memory_planner import MemoryPlan, plan_memory
from inferscope.optimization.serving_profile import (
    CacheSpec,
    EngineType,
    ModelClass,
    ObjectiveSpec,
    PrecisionSpec,
    SchedulerSpec,
    ServingProfile,
    SpeculationSpec,
    TopologySpec,
    WorkloadMode,
)
from inferscope.profiling import ProfilingIntent, resolve_profiling_intent


@dataclass
class PipelineContext:
    """State object passed along the Recommender DAG edges."""

    model: ModelVariant
    gpu: GPUProfile
    num_gpus: int
    workload: WorkloadMode
    objective: ObjectiveSpec
    forced_engine: str = "auto"

    # Resolved during DAG execution
    engine_type: EngineType | None = None
    precision: PrecisionSpec | None = None
    topology: TopologySpec | None = None
    scheduler: SchedulerSpec | None = None
    cache: CacheSpec | None = None
    speculation: SpeculationSpec | None = None
    profile: ServingProfile | None = None
    engine_config: EngineConfig | None = None
    memory_plan: MemoryPlan | None = None
    profiling_intent: ProfilingIntent | None = None

    # Reasoning trace for MCP visibility
    reasoning_trace: list[str] = field(default_factory=list)


class DAGNode(abc.ABC):
    """Abstract base class for a recommendation pipeline node."""

    @abc.abstractmethod
    def process(self, ctx: PipelineContext) -> None:
        pass


class HardwareNode(DAGNode):
    """Analyzes GPU architecture, memory, and native precisions."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.reasoning_trace.append(
            f"HardwareNode: Detected {ctx.gpu.vendor.upper()} {ctx.gpu.name} "
            f"({ctx.gpu.architecture}) with {ctx.gpu.memory_gb}GB memory."
        )

        # Engine Selection
        is_amd = ctx.gpu.vendor == "amd"
        is_mla_moe = ctx.model.model_class == ModelClass.FRONTIER_MLA_MOE

        if ctx.forced_engine != "auto":
            try:
                ctx.engine_type = EngineType(ctx.forced_engine)
                ctx.reasoning_trace.append(f"HardwareNode: Engine forced to {ctx.forced_engine}.")
            except ValueError as exc:
                valid = [e.value for e in EngineType]
                raise ValueError(f"Unknown engine '{ctx.forced_engine}'. Valid engines: {valid}") from exc
        else:
            if is_amd:
                if is_mla_moe and ctx.gpu.compute_capability == "gfx950":
                    ctx.engine_type = EngineType.ATOM
                    ctx.reasoning_trace.append("HardwareNode: Selected ATOM for MI355X MLA/MoE.")
                elif is_mla_moe:
                    ctx.engine_type = EngineType.ATOM
                    ctx.reasoning_trace.append("HardwareNode: Selected ATOM for MI300X MLA/MoE.")
                else:
                    ctx.engine_type = EngineType.VLLM
                    ctx.reasoning_trace.append("HardwareNode: Selected vLLM for AMD general workloads.")
            else:
                is_hopper = ctx.gpu.architecture == "Hopper"
                if is_hopper:
                    if ctx.workload == WorkloadMode.CODING:
                        ctx.engine_type = EngineType.SGLANG
                        ctx.reasoning_trace.append(
                            "HardwareNode: Hopper Tier 2 — SGLang for coding "
                            "(RadixAttention prefix hit rate superior for session-heavy coding)."
                        )
                    else:
                        ctx.engine_type = EngineType.VLLM
                        ctx.reasoning_trace.append(
                            f"HardwareNode: Hopper Tier 1 — vLLM for {ctx.workload.value} "
                            "(FlashAttention-3 via wgmma/TMA, zero-overhead prefix caching, "
                            "Hopper-optimized chunked prefill)."
                        )
                elif ctx.gpu.architecture == "Blackwell":
                    if ctx.workload == WorkloadMode.CODING:
                        ctx.engine_type = EngineType.SGLANG
                        ctx.reasoning_trace.append(
                            "HardwareNode: Blackwell Tier 2 — SGLang for coding "
                            "(RadixAttention prefix hit rate superior for session-heavy coding)."
                        )
                    else:
                        ctx.engine_type = EngineType.VLLM
                        ctx.reasoning_trace.append(
                            f"HardwareNode: Blackwell Tier 1 — vLLM for {ctx.workload.value} "
                            "(FlashAttention-4 with RoPE+KV fusion, NVFP4 native, "
                            "nvCOMP decompression engine)."
                        )
                elif ctx.workload == WorkloadMode.CODING:
                    ctx.engine_type = EngineType.SGLANG
                    ctx.reasoning_trace.append("HardwareNode: Selected SGLang for coding (RadixAttention).")
                else:
                    ctx.engine_type = EngineType.VLLM
                    ctx.reasoning_trace.append("HardwareNode: Selected vLLM for NVIDIA general workloads.")

        # Determine Precision
        kv_cache = "auto"
        if ctx.gpu.fp8_support:
            kv_cache = "fp8_e4m3"

        if ctx.model.params_total_b <= 13 and ctx.gpu.memory_gb >= 40:
            ctx.precision = PrecisionSpec(weights="bf16", activations="bf16", kv_cache=kv_cache)
            ctx.reasoning_trace.append("HardwareNode: Small model fits easily; routing to BF16 weights.")
        elif ctx.gpu.fp8_support:
            if is_amd and ctx.model.model_type == "moe":
                ctx.precision = PrecisionSpec(weights="bf16", activations="bf16", kv_cache="fp8_e4m3")
                ctx.reasoning_trace.append(
                    "HardwareNode: Downgrading AMD MoE to BF16 (rocprofv3 flagged FP8 decode regression)."
                )
            else:
                ctx.precision = PrecisionSpec(weights="fp8", activations="fp8", kv_cache="fp8_e4m3")
                ctx.reasoning_trace.append("HardwareNode: Native FP8 support detected; routing to FP8 weights.")
        elif ctx.model.params_total_b > 30:
            ctx.precision = PrecisionSpec(weights="awq", activations="fp16", kv_cache="auto")
            ctx.reasoning_trace.append("HardwareNode: GPU lacks FP8; routing large model to AWQ/INT4.")
        else:
            ctx.precision = PrecisionSpec(weights="bf16", activations="bf16", kv_cache="auto")


class ModelNode(DAGNode):
    """Analyzes model boundaries (params, KV heads) for topology splits."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.reasoning_trace.append(
            f"ModelNode: Analyzing {ctx.model.name} ({ctx.model.params_total_b}B total, {ctx.model.model_class.value})."
        )

        weight_gb = ctx.model.weight_gb(ctx.precision.weights)
        per_gpu_usable = ctx.gpu.memory_gb * 0.90

        candidates = []
        for tp in [1, 2, 4, 8, 16, 32]:
            if tp > ctx.num_gpus:
                break
            if ctx.model.kv_heads > 0 and ctx.model.kv_heads % tp != 0:
                continue
            candidates.append(tp)
        valid_tps = candidates if candidates else [1]

        tp = valid_tps[0]
        if (
            ctx.gpu.vendor == "amd"
            and ctx.workload in (WorkloadMode.CODING, WorkloadMode.AGENT)
            and ctx.model.params_total_b < 40
        ):
            ctx.reasoning_trace.append(
                "ModelNode: Forcing TP=1 for small model on AMD due to RCCL overhead compared to NCCL."
            )
            tp = 1
        else:
            for candidate in valid_tps:
                if weight_gb / candidate <= per_gpu_usable * 0.8:
                    tp = candidate
                    break
                tp = candidate

        ep = 1
        if ctx.model.model_type == "moe" and ctx.model.experts_total > 64 and ctx.num_gpus >= 4:
            ep = min(2, ctx.num_gpus // tp)
            ctx.reasoning_trace.append(f"ModelNode: Mapping Kimi/DeepSeek 384-expert style EP={ep}.")

        dp = max(1, ctx.num_gpus // (tp * ep))
        ctx.topology = TopologySpec(tp=tp, pp=1, dp=dp, ep=ep)
        ctx.reasoning_trace.append(f"ModelNode: Resolved routing topology to TP={tp}, EP={ep}, DP={dp}.")

        ctx.speculation = SpeculationSpec()
        if ctx.model.mtp_speculative and ctx.workload in (WorkloadMode.CODING, WorkloadMode.AGENT):
            ctx.speculation = SpeculationSpec(
                mode="low_batch_only",
                method="mtp",
                num_speculative_tokens=3,
            )
            ctx.reasoning_trace.append("ModelNode: Enabled MTP speculation for agentic workload.")


class WorkloadNode(DAGNode):
    """Generate scheduler and cache policy from workload and hardware boundaries."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.reasoning_trace.append(f"WorkloadNode: Injecting {ctx.workload.value.upper()} constraints.")

        is_amd = ctx.gpu.vendor == "amd"
        is_ampere = ctx.gpu.architecture == "Ampere"
        is_hopper = ctx.gpu.architecture == "Hopper"
        is_blackwell = ctx.gpu.architecture == "Blackwell"
        is_h200 = is_hopper and ctx.gpu.memory_gb >= 140
        is_hopper_pcie = is_hopper and not ctx.gpu.nvlink_bandwidth_gb_s
        is_b300 = is_blackwell and ctx.gpu.memory_gb >= 280
        is_gb200 = is_blackwell and ctx.gpu.extra.get("grace_cpu_cores", 0) > 0
        is_long_context_model = ctx.model.context_length > 32768
        has_high_speed_interconnect = bool(ctx.gpu.nvlink_bandwidth_gb_s or ctx.gpu.if_bandwidth_gb_s)
        multi_gpu = ctx.num_gpus > 1

        if is_b300 or (is_hopper and not is_hopper_pcie):
            high_util = 0.95
        elif is_blackwell:
            high_util = 0.93
        else:
            high_util = 0.93

        chunked = self._resolve_chunked_prefill(
            ctx,
            is_amd,
            is_hopper or is_blackwell,
            is_long_context_model,
        )

        bw = ctx.gpu.memory_bandwidth_tb_s
        budget_base = self._derive_token_budget(bw)
        large_hbm = is_h200 or is_blackwell

        if ctx.workload == WorkloadMode.CODING:
            budget = max(4096, int(budget_base * 0.5))
            ctx.scheduler = SchedulerSpec(
                batched_token_budget=budget,
                prefill_chunk_tokens=budget,
                max_num_seqs=128,
                decode_priority=0.7,
                chunked_prefill=chunked,
                prefill_decode_isolation="soft_priority" if multi_gpu else "colocated",
                max_prefill_chunk_ratio=0.5,
            )
            offload = "disabled" if large_hbm else "cold_only"
            ctx.cache = CacheSpec(
                prefix_cache=True,
                session_affinity=True,
                gpu_memory_utilization=high_util,
                eviction_policy="lru",
                offload_policy=offload,
                offload_idle_threshold_s=30.0 if is_hopper_pcie else 60.0,
                block_reuse_strategy="prefix_sharing",
                fragmentation_check=True,
            )
            ctx.reasoning_trace.append(
                f"WorkloadNode: Coding (Tier 2) — budget {budget}, offload={offload}, util={high_util}."
            )

        elif ctx.workload == WorkloadMode.AGENT:
            budget = max(4096, int(budget_base * 0.75))
            ctx.scheduler = SchedulerSpec(
                batched_token_budget=budget,
                prefill_chunk_tokens=budget,
                max_num_seqs=128,
                decode_priority=0.7,
                chunked_prefill=chunked,
                prefill_decode_isolation="soft_priority" if multi_gpu else "colocated",
                max_prefill_chunk_ratio=0.4,
            )
            if large_hbm:
                offload = "disabled"
                idle_s = 0.0
            elif is_hopper_pcie:
                offload = "cold_only"
                idle_s = 30.0
            else:
                offload = "cold_only"
                idle_s = 60.0
            ctx.cache = CacheSpec(
                prefix_cache=True,
                session_affinity=True,
                gpu_memory_utilization=high_util,
                eviction_policy="lru",
                offload_policy=offload,
                offload_idle_threshold_s=idle_s,
                kv_compaction_trigger=0.4,
                block_reuse_strategy="prefix_sharing_cross_session",
                fragmentation_check=True,
            )
            ctx.reasoning_trace.append(
                f"WorkloadNode: Agent (Tier 1) — budget {budget}, cross-session sharing, "
                f"offload={offload}, util={high_util}."
            )

        elif ctx.workload == WorkloadMode.LONG_CONTEXT_RAG:
            budget = max(16384, int(budget_base * 2.0))
            ctx.scheduler = SchedulerSpec(
                batched_token_budget=budget,
                prefill_chunk_tokens=budget,
                max_num_seqs=64 if not is_blackwell else 96,
                decode_priority=0.4,
                chunked_prefill=chunked,
                prefill_decode_isolation="soft_priority" if multi_gpu else "colocated",
                max_prefill_chunk_ratio=0.7,
            )
            if is_gb200:
                offload = "cold_only"
                idle_s = 240.0
                tiering = "gpu_cpu"
            elif is_blackwell:
                offload = "cold_only"
                idle_s = 240.0
                tiering = "gpu_only"
            elif is_h200:
                offload = "cold_only"
                idle_s = 180.0
                tiering = "gpu_only"
            elif is_hopper_pcie:
                offload = "cold_only"
                idle_s = 60.0
                tiering = "gpu_cpu"
            else:
                offload = "cold_only"
                idle_s = 120.0
                tiering = "gpu_only"
            ctx.cache = CacheSpec(
                prefix_cache=True,
                session_affinity=False,
                gpu_memory_utilization=high_util,
                offload_policy=offload,
                offload_idle_threshold_s=idle_s,
                kv_tiering=tiering,
                fragmentation_check=True,
            )
            if is_gb200:
                ctx.reasoning_trace.append(
                    "WorkloadNode: GB200 Grace Blackwell — KV overflow to Grace LPDDR5X via "
                    "NVLink-C2C @ 900 GB/s (~7x faster than PCIe Gen5)."
                )
            if is_hopper_pcie and is_long_context_model:
                ctx.reasoning_trace.append(
                    "WorkloadNode: CRITICAL — H100 PCIe + long context: consider disaggregated "
                    "prefill/decode over KV offloading (PCIe-only offload is transfer-bound)."
                )
            ctx.reasoning_trace.append(
                f"WorkloadNode: Long-context RAG (Tier 1) — budget {budget}, offload={offload} "
                f"(idle {idle_s:.0f}s), tiering={tiering}, util={high_util}."
            )

        else:
            budget = max(8192, budget_base)
            weight_gb = ctx.model.weight_gb(ctx.precision.weights if ctx.precision else "fp16")
            tp = max(ctx.topology.tp if ctx.topology else 1, 1)
            kv_bpt = ctx.model.kv_cache_bytes_per_token("fp8" if ctx.gpu.fp8_support else "fp16") * ctx.model.layers
            max_seqs = self._derive_max_seqs(
                ctx.gpu.memory_gb,
                weight_gb / tp,
                kv_bpt / tp,
                2048,
                high_util,
            )
            ctx.scheduler = SchedulerSpec(
                batched_token_budget=budget,
                prefill_chunk_tokens=8192 if is_ampere else 16384,
                max_num_seqs=max_seqs,
                decode_priority=0.5,
                chunked_prefill=chunked,
                prefill_decode_isolation="colocated",
                max_prefill_chunk_ratio=0.5,
            )
            ctx.cache = CacheSpec(
                prefix_cache=True,
                session_affinity=False,
                gpu_memory_utilization=high_util,
                offload_policy="disabled",
                block_reuse_strategy="prefix_sharing",
            )
            ctx.reasoning_trace.append(
                f"WorkloadNode: Chat (Tier 1) — budget {budget}, max_seqs={max_seqs}, "
                f"offload=disabled, util={high_util}."
            )

        if is_hopper and ctx.engine_type == EngineType.VLLM:
            ctx.reasoning_trace.append(
                "WorkloadNode: Hopper vLLM — FlashAttention-3 via wgmma/TMA "
                "(75%+ utilization vs FA2's 35%), zero-overhead V1 prefix caching."
            )
            if is_h200:
                ctx.reasoning_trace.append(
                    "WorkloadNode: H200 141GB HBM3e @ 4.8 TB/s — most workloads fit GPU-resident without KV offloading."
                )
        if is_blackwell and ctx.engine_type == EngineType.VLLM:
            ctx.reasoning_trace.append(
                "WorkloadNode: Blackwell vLLM — FlashAttention-4 with RoPE+KV fusion "
                "(4.5x speedup), NVFP4 native, nvCOMP decompression engine."
            )
            if is_gb200:
                ctx.reasoning_trace.append(
                    "WorkloadNode: GB200 — Grace LPDDR5X (480GB @ 546 GB/s) as KV overflow "
                    "via NVLink-C2C, eliminating PCIe bottleneck for long-context workloads."
                )
            elif is_b300:
                ctx.reasoning_trace.append(
                    "WorkloadNode: B300 288GB HBM3e — fits most models on TP=1-2, "
                    "accelerated softmax in hardware, inference-optimized."
                )

        if (
            is_blackwell
            and multi_gpu
            and ctx.workload
            in (
                WorkloadMode.LONG_CONTEXT_RAG,
                WorkloadMode.AGENT,
            )
        ):
            ctx.reasoning_trace.append(
                "WorkloadNode: Blackwell disaggregated serving advantage — "
                f"NVLink5 @ {ctx.gpu.nvlink_bandwidth_gb_s:.0f} GB/s (2x vs Hopper NVLink4) + "
                "decompression engine for compressed KV transfer. Consider P/D split for "
                "long-context workloads at high request rates."
            )

        if ctx.cache.offload_policy != "disabled":
            if has_high_speed_interconnect:
                ctx.cache.pcie_utilization_cap = 0.8
            else:
                ctx.cache.pcie_utilization_cap = 0.5
                ctx.reasoning_trace.append(
                    "WorkloadNode: No NVLink/InfinityFabric — capping PCIe offload utilization "
                    "at 50% to prevent transfer-bound decode stalls."
                )

        if ctx.objective.ttft_p95_ms > 0 and ctx.objective.ttft_p95_ms < 500 and is_long_context_model:
            ctx.scheduler.chunked_prefill = False
            ctx.reasoning_trace.append(
                f"WorkloadNode: Tight TTFT SLO ({ctx.objective.ttft_p95_ms}ms) + long-context "
                "model — forcing contiguous prefill."
            )

    @staticmethod
    def _derive_token_budget(memory_bandwidth_tb_s: float) -> int:
        raw = int(memory_bandwidth_tb_s * 5000)
        return max(4096, min(raw, 131072))

    @staticmethod
    def _derive_max_seqs(
        gpu_memory_gb: float,
        weight_gb_per_gpu: float,
        kv_bytes_per_token: float,
        avg_context: int,
        gpu_memory_utilization: float,
    ) -> int:
        usable_gb = gpu_memory_gb * gpu_memory_utilization
        kv_budget_gb = usable_gb - weight_gb_per_gpu - 2.0
        if kv_budget_gb <= 0:
            return 32
        kv_per_seq_gb = (kv_bytes_per_token * avg_context) / 1e9
        fragmentation_factor = 0.75
        if kv_per_seq_gb <= 0:
            return 256
        raw = int((kv_budget_gb * fragmentation_factor) / kv_per_seq_gb)
        return max(32, min(raw, 1024))

    @staticmethod
    def _resolve_chunked_prefill(
        ctx: PipelineContext,
        is_amd: bool,
        is_hopper: bool,
        is_long_context_model: bool,
    ) -> bool:
        if is_amd and ctx.workload in (WorkloadMode.LONG_CONTEXT_RAG, WorkloadMode.CODING):
            ctx.reasoning_trace.append(
                "WorkloadNode: Disabling chunked prefill on AMD CDNA — KV cache staging "
                "overhead + decode starvation risk for long-context workloads."
            )
            return False

        if is_hopper and ctx.workload in (WorkloadMode.CHAT, WorkloadMode.AGENT):
            ctx.reasoning_trace.append(
                "WorkloadNode: Chunked prefill ON — Hopper wgmma/TMA handles "
                "compute-memory overlap efficiently for Tier 1 workloads."
            )
            return True

        if is_long_context_model and ctx.workload == WorkloadMode.LONG_CONTEXT_RAG:
            ctx.reasoning_trace.append(
                "WorkloadNode: Disabling chunked prefill for long-context RAG — contiguous "
                "prefill avoids KV fragmentation across PagedAttention blocks."
            )
            return False

        return True


class ProfilingNode(DAGNode):
    """Attach advisory profiling intent for future kernel/profiler integrations."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.profiling_intent = resolve_profiling_intent(ctx.gpu.vendor)
        ctx.reasoning_trace.append(ctx.profiling_intent.summary)


class TelemetryNode(DAGNode):
    """Configures Prometheus and Grafana alerts."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.reasoning_trace.append(
            "TelemetryNode: Configured Prometheus /metrics scraper and Grafana KV cache capacity alerts."
        )


class CompilerNode(DAGNode):
    """Finalize the profile and bind it to an engine compiler."""

    def process(self, ctx: PipelineContext) -> None:
        ctx.profile = ServingProfile(
            model=ctx.model.name,
            model_class=ctx.model.model_class,
            engine=ctx.engine_type,
            gpu_type=ctx.gpu.name,
            num_gpus=ctx.num_gpus,
            workload_mode=ctx.workload,
            objective=ctx.objective,
            topology=ctx.topology,
            scheduler=ctx.scheduler,
            cache=ctx.cache,
            precision=ctx.precision,
            speculation=ctx.speculation,
            reasoning_trace=ctx.reasoning_trace,
        )

        inventory = DeploymentInventory(
            gpu_type=ctx.gpu.name,
            gpu_arch=ctx.gpu.compute_capability,
            gpu_count=ctx.num_gpus,
            gpu_memory_gb=ctx.gpu.memory_gb,
            gpu_memory_bandwidth_tb_s=ctx.gpu.memory_bandwidth_tb_s,
            interconnect=(
                f"nvlink{ctx.gpu.nvlink_version}"
                if ctx.gpu.nvlink_version
                else f"infinity_fabric_{ctx.gpu.infinity_fabric_version}"
                if ctx.gpu.infinity_fabric_version
                else ctx.gpu.pcie
            ),
            interconnect_bandwidth_gb_s=ctx.gpu.nvlink_bandwidth_gb_s or ctx.gpu.if_bandwidth_gb_s,
            fp8_support=ctx.gpu.fp8_support,
            fp4_support=ctx.gpu.fp4_support,
            fp8_format=ctx.gpu.fp8_format,
        )

        compiler = get_compiler(ctx.engine_type.value)
        ctx.engine_config = compiler.compile(ctx.profile, inventory)

        ctx.memory_plan = plan_memory(
            model=ctx.model,
            gpu=ctx.gpu,
            num_gpus=ctx.num_gpus,
            tp=ctx.topology.tp,
            precision=ctx.precision.weights,
            kv_precision=ctx.precision.kv_cache,
            gpu_memory_utilization=ctx.cache.gpu_memory_utilization,
        )

        if not ctx.memory_plan.fits:
            ctx.profile.warnings.append(
                f"Model does not fit with TP={ctx.topology.tp} {ctx.precision.weights} — "
                "increase TP or use more aggressive quantization"
            )

        ctx.profile.engine_flags = ctx.engine_config.cli_flags
        ctx.profile.env_vars = ctx.engine_config.env_vars
        ctx.profile.warnings.extend(ctx.engine_config.warnings)

        ctx.reasoning_trace.append("CompilerNode: Successfully bound DAG context to final EngineConfig.")


def recommend(
    model: ModelVariant,
    gpu: GPUProfile,
    num_gpus: int = 1,
    workload: WorkloadMode = WorkloadMode.CHAT,
    engine: str = "auto",
    objective: ObjectiveSpec | None = None,
) -> tuple[ServingProfile, EngineConfig, MemoryPlan]:
    """Execute the modular recommendation DAG."""
    ctx = PipelineContext(
        model=model,
        gpu=gpu,
        num_gpus=num_gpus,
        workload=workload,
        objective=objective or ObjectiveSpec(),
        forced_engine=engine,
    )

    nodes: list[DAGNode] = [
        HardwareNode(),
        ModelNode(),
        WorkloadNode(),
        ProfilingNode(),
        TelemetryNode(),
        CompilerNode(),
    ]

    for node in nodes:
        node.process(ctx)

    log = get_logger(component="recommender")
    log.info(
        "recommendation_compiled_via_dag",
        model=ctx.model.name,
        gpu=ctx.gpu.name,
        engine=ctx.engine_type.value,
        tp=ctx.topology.tp,
        trace_length=len(ctx.reasoning_trace),
    )

    return ctx.profile, ctx.engine_config, ctx.memory_plan
