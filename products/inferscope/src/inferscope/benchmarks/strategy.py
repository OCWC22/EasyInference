"""Benchmark strategy and benchmark-to-profile bridge for InferScope."""

from __future__ import annotations

from typing import Any

from inferscope.benchmarks.catalog import (
    build_benchmark_matrix,
    describe_builtin_experiments,
    describe_builtin_workloads,
    load_experiment,
    load_workload,
)
from inferscope.benchmarks.launchers import build_benchmark_stack_plan
from inferscope.benchmarks.support import assess_benchmark_support
from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.platform_policy import resolve_platform_traits
from inferscope.optimization.serving_profile import WorkloadMode
from inferscope.tools.kv_cache import recommend_disaggregation, recommend_kv_strategy
from inferscope.tools.profiling import profile_runtime
from inferscope.tools.recommend import recommend_config


def _normalize_workload_mode(value: str) -> WorkloadMode:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "coding_agent": WorkloadMode.CODING,
        "tool_agent": WorkloadMode.AGENT,
        "agent": WorkloadMode.AGENT,
        "chat": WorkloadMode.CHAT,
        "reasoning_chat": WorkloadMode.CHAT,
        "rag": WorkloadMode.LONG_CONTEXT_RAG,
        "long_context_rag": WorkloadMode.LONG_CONTEXT_RAG,
    }
    if normalized in aliases:
        return aliases[normalized]
    return WorkloadMode(normalized)


def _focus_area_for_mode(mode: WorkloadMode) -> str:
    if mode == WorkloadMode.AGENT:
        return "tool_calling"
    if mode == WorkloadMode.CODING:
        return "long_context"
    if mode == WorkloadMode.LONG_CONTEXT_RAG:
        return "kv_offload"
    return "reasoning"


def _primary_workload_name(
    mode: WorkloadMode,
    *,
    avg_prompt_tokens: int,
    concurrent_sessions: int,
) -> str:
    if mode == WorkloadMode.AGENT:
        return "tool-agent"
    if mode == WorkloadMode.CODING:
        if avg_prompt_tokens >= 16_384 and concurrent_sessions >= 4:
            return "enterprise-coding-agent"
        return "coding-long-context"
    if mode == WorkloadMode.LONG_CONTEXT_RAG:
        return "long-context-kv-offload-rag"
    return "reasoning-chatbot"


def _supplemental_workload_names(mode: WorkloadMode, primary_workload: str) -> list[str]:
    candidates = {
        WorkloadMode.CODING: ["enterprise-coding-agent", "disagg-long-prompt"],
        WorkloadMode.AGENT: ["coding-long-context"],
        WorkloadMode.LONG_CONTEXT_RAG: ["medical-rag", "legal-review"],
        WorkloadMode.CHAT: ["neo-cloud-mixed"],
    }.get(mode, [])
    return [candidate for candidate in candidates if candidate != primary_workload]


def _find_descriptor(descriptors: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((descriptor for descriptor in descriptors if descriptor["name"] == name), None)


def _make_lane(
    *,
    phase: str,
    objective: str,
    experiment: str,
    workload: str,
    rationale: str,
    required: bool = True,
) -> dict[str, Any]:
    descriptor = _find_descriptor(describe_builtin_experiments(), experiment)
    if descriptor is None:
        raise ValueError(f"Unknown experiment '{experiment}'")
    return {
        "phase": phase,
        "objective": objective,
        "required": required,
        "experiment": experiment,
        "workload": workload,
        "engine": descriptor["engine"],
        "topology_mode": descriptor["topology_mode"],
        "cache_strategy": descriptor["cache_strategy"],
        "focus_areas": descriptor["focus_areas"],
        "benchmark_role": descriptor["benchmark_role"],
        "rationale": rationale,
    }


def _select_suite_lanes(
    *,
    workload_mode: WorkloadMode,
    primary_workload: str,
    selected_engine: str,
    num_gpus: int,
    avg_prompt_tokens: int,
    disaggregation: dict[str, Any],
    has_grace: bool,
    has_rdma: bool,
    is_nvidia: bool = False,
) -> list[dict[str, Any]]:
    lanes: list[dict[str, Any]] = []

    if workload_mode == WorkloadMode.LONG_CONTEXT_RAG:
        lanes.append(
            _make_lane(
                phase="reference",
                objective="Measure GPU-resident long-context baseline before any KV overflow strategy.",
                experiment="vllm-single-endpoint-long-context-rag-baseline",
                workload=primary_workload,
                rationale="Start with a clean single-endpoint vLLM baseline for the exact long-context RAG workload.",
            )
        )
        lanes.append(
            _make_lane(
                phase="offload",
                objective="Measure cold-session host spill and session reuse under realistic KV pressure.",
                experiment="vllm-single-endpoint-offloading-connector",
                workload=primary_workload,
                rationale=(
                    "Use this as the single-endpoint host-DRAM control lane before comparing "
                    "against remote or Grace-coherent overflow paths."
                    if has_grace
                    else "This isolates explicit host-memory overflow without changing deployment topology."
                ),
            )
        )
        if num_gpus >= 2:
            # Dynamo + NIXL is the production disaggregated lane
            dynamo_experiment = (
                "dynamo-disagg-prefill-nixl-grace" if has_grace else "dynamo-disagg-prefill-nixl-rag"
            )
            lanes.append(
                _make_lane(
                    phase="disaggregated",
                    objective="Measure Dynamo-orchestrated prefill/decode split with NIXL KV transfer for long-context serving.",
                    experiment=dynamo_experiment,
                    workload=primary_workload,
                    rationale=(
                        "Dynamo 1.0 is the production orchestration layer for disaggregated NVIDIA deployments. "
                        + ("Grace platforms benchmark coherent overflow explicitly."
                           if has_grace
                           else "NIXL provides zero-copy RDMA KV transfer between prefill and decode pools.")
                    ),
                    required=bool(disaggregation.get("recommended", False)),
                )
            )
            # Keep LMCache as a secondary comparison lane
            lmcache_experiment = "vllm-disagg-prefill-lmcache-grace" if has_grace else "vllm-disagg-prefill-lmcache-rag"
            lanes.append(
                _make_lane(
                    phase="cache_extension",
                    objective="Compare raw vLLM worker disaggregation via LMCache against Dynamo-orchestrated path.",
                    experiment=lmcache_experiment,
                    workload=primary_workload,
                    rationale=(
                        "LMCache lane provides a direct worker-only comparison without Dynamo orchestration overhead."
                    ),
                    required=False,
                )
            )
        return lanes

    if workload_mode == WorkloadMode.AGENT:
        lanes.append(
            _make_lane(
                phase="reference",
                objective="Measure session reuse and prefix-aware agent performance on a single endpoint.",
                experiment="sglang-single-endpoint-hicache",
                workload="tool-agent",
                rationale="InferScope's MCP-style agent lane is currently best represented by SGLang HiCache.",
            )
        )
        if num_gpus >= 2:
            lanes.append(
                _make_lane(
                    phase="disaggregated",
                    objective="Measure routed agent sessions across router, prefill, and decode roles.",
                    experiment="sglang-router-prefill-decode",
                    workload="tool-agent",
                    rationale="This isolates routing, sticky-session behavior, and cache-aware multi-role serving.",
                    required=selected_engine == "sglang",
                )
            )
        return lanes

    if workload_mode == WorkloadMode.CODING:
        lanes.append(
            _make_lane(
                phase="reference",
                objective="Measure GPU-resident long-context coding baseline with prefix caching.",
                experiment="vllm-single-endpoint-baseline",
                workload="coding-long-context",
                rationale="This is the canonical coding baseline aligned with InferScope's optimizer path.",
            )
        )
        if selected_engine == "sglang":
            lanes.append(
                _make_lane(
                    phase="cache_extension",
                    objective="Measure longest-prefix-match behavior on coding/agent-style reuse.",
                    experiment="sglang-single-endpoint-hicache",
                    workload="tool-agent",
                    rationale=(
                        "Even for coding deployments, this is the best packaged proxy for SGLang cache reuse behavior."
                    ),
                    required=False,
                )
            )
            if num_gpus >= 2:
                lanes.append(
                    _make_lane(
                        phase="disaggregated",
                        objective="Measure routed multi-role serving for prefix-heavy interactive workflows.",
                        experiment="sglang-router-prefill-decode",
                        workload="tool-agent",
                        rationale=(
                            "Use SGLang's routed lane when the deployment target is "
                            "SGLang and multi-node planning matters."
                        ),
                        required=avg_prompt_tokens >= 8_192,
                    )
                )
                # Also add Dynamo disaggregated lane for NVIDIA platforms as a comparison
                if is_nvidia:
                    dynamo_experiment = (
                        "dynamo-disagg-prefill-nixl-grace" if has_grace else "dynamo-disagg-prefill-nixl"
                    )
                    lanes.append(
                        _make_lane(
                            phase="disaggregated",
                            objective="Compare Dynamo-orchestrated NIXL disaggregation against SGLang routing for coding.",
                            experiment=dynamo_experiment,
                            workload="coding-long-context",
                            rationale=(
                                "Dynamo 1.0 provides production KV-aware routing with NIXL transfer — "
                                "compare against SGLang's native routing to determine the optimal NVIDIA path."
                            ),
                            required=False,
                        )
                    )
            return lanes

        if num_gpus >= 2:
            if is_nvidia:
                # Dynamo is the production disaggregated lane for NVIDIA coding workloads
                dynamo_experiment = (
                    "dynamo-disagg-prefill-nixl-grace" if has_grace else "dynamo-disagg-prefill-nixl"
                )
                lanes.append(
                    _make_lane(
                        phase="disaggregated",
                        objective="Measure Dynamo-orchestrated prefill/decode split with NIXL KV transfer for coding.",
                        experiment=dynamo_experiment,
                        workload="coding-long-context",
                        rationale=(
                            "Dynamo 1.0 is the production orchestration layer for NVIDIA disaggregated serving "
                            "with KV-aware routing and NIXL transfer."
                        ),
                        required=bool(disaggregation.get("recommended", False)),
                    )
                )
            elif has_rdma:
                lanes.append(
                    _make_lane(
                        phase="disaggregated",
                        objective="Measure high-speed KV transfer for long-prompt coding traffic.",
                        experiment="vllm-disagg-prefill-nixl",
                        workload="coding-long-context",
                        rationale=(
                            "Prefer NIXL when RDMA or fast transport is available "
                            "and prompt transfer is the main concern."
                        ),
                        required=bool(disaggregation.get("recommended", False)),
                    )
                )
            lanes.append(
                _make_lane(
                    phase="cache_extension",
                    objective="Compare raw vLLM worker disaggregation via LMCache against the primary disaggregated path.",
                    experiment="vllm-disagg-prefill-lmcache",
                    workload="coding-long-context",
                    rationale=(
                        "LMCache lane provides a direct worker-only comparison for content-addressed KV reuse."
                    ),
                    required=not is_nvidia and avg_prompt_tokens >= 8_192,
                )
            )
        return lanes

    return lanes


def _lane_priority(
    lane: dict[str, Any],
    *,
    runtime_profile: dict[str, Any] | None,
    disaggregation: dict[str, Any],
) -> int:
    priority = {"reference": 10, "cache_extension": 20, "offload": 30, "disaggregated": 40}.get(lane["phase"], 50)
    if runtime_profile is None:
        return priority

    bottleneck_kinds = {
        bottleneck.get("kind", "")
        for bottleneck in runtime_profile.get("bottlenecks", [])
        if isinstance(bottleneck, dict)
    }
    memory_level = str(runtime_profile.get("memory_pressure", {}).get("level", ""))
    cache_effectiveness = str(runtime_profile.get("cache_effectiveness", {}).get("effectiveness", ""))

    if lane["phase"] == "offload" and ("cache_bound" in bottleneck_kinds or memory_level in {"high", "critical"}):
        priority -= 25
    if lane["phase"] == "disaggregated" and (
        "prefill_compute_bound" in bottleneck_kinds or disaggregation.get("recommended", False)
    ):
        priority -= 20
    if lane["phase"] == "cache_extension" and cache_effectiveness in {"poor", "minimal", "disabled_or_no_data"}:
        priority -= 10
    return priority


def _next_actions(
    lanes: list[dict[str, Any]],
    *,
    runtime_profile: dict[str, Any] | None,
) -> list[str]:
    if runtime_profile is None:
        return [
            f"Run the {lane['experiment']} lane to validate: {lane['objective']}"
            for lane in lanes[:3]
            if lane.get("required", True)
        ]

    actions = [
        f"Benchmark next with {lane['experiment']} ({lane['phase']}) — {lane['objective']}" for lane in lanes[:2]
    ]
    tuning_preview = runtime_profile.get("tuning_preview") or {}
    adjustments = tuning_preview.get("adjustments") or []
    for adjustment in adjustments[:2]:
        if isinstance(adjustment, dict):
            actions.append(
                f"Validate tuning hypothesis: set {adjustment['parameter']} to {adjustment['recommended_value']}"
            )
    return actions


def plan_benchmark_strategy(
    model: str,
    gpu: str,
    *,
    workload: str = "chat",
    num_gpus: int = 1,
    engine: str = "auto",
    max_context: int = 32_768,
    concurrent_sessions: int = 100,
    avg_prompt_tokens: int = 4_096,
    request_rate_per_sec: float = 10.0,
    has_rdma: bool = False,
    multi_node: bool = False,
    host: str = "127.0.0.1",
    include_stack_plans: bool = True,
) -> dict[str, Any]:
    """Plan the right benchmark suite for this deployment target."""
    model_variant = get_model_variant(model)
    if model_variant is None:
        return {"error": f"Unknown model: '{model}'", "confidence": 0.0}
    gpu_profile = get_gpu_profile(gpu)
    if gpu_profile is None:
        return {"error": f"Unknown GPU: '{gpu}'", "confidence": 0.0}

    try:
        workload_mode = _normalize_workload_mode(workload)
    except ValueError:
        return {"error": f"Unknown workload: '{workload}'", "confidence": 0.0}

    recommendation = recommend_config(model, gpu, workload_mode.value, num_gpus, engine=engine)
    if "error" in recommendation:
        return recommendation
    selected_engine = str(recommendation["engine_config"]["engine"])

    kv_strategy = recommend_kv_strategy(
        model,
        gpu,
        workload_mode.value,
        max_context=max_context,
        concurrent_sessions=concurrent_sessions,
    )
    disagg = recommend_disaggregation(
        model,
        gpu,
        avg_prompt_tokens=avg_prompt_tokens,
        request_rate_per_sec=request_rate_per_sec,
        has_rdma=has_rdma,
        num_gpus=num_gpus,
    )
    traits = resolve_platform_traits(gpu_profile)
    primary_workload = _primary_workload_name(
        workload_mode,
        avg_prompt_tokens=avg_prompt_tokens,
        concurrent_sessions=concurrent_sessions,
    )
    supplemental_workloads = _supplemental_workload_names(workload_mode, primary_workload)
    focus_area = _focus_area_for_mode(workload_mode)

    matrix = build_benchmark_matrix(
        gpu_family=traits.family.value,
        model_class=model_variant.model_class.value,
        focus_area=focus_area,
    )
    workload_descriptors = describe_builtin_workloads()
    primary_descriptor = _find_descriptor(workload_descriptors, primary_workload)
    supplemental_descriptors = [
        descriptor for descriptor in workload_descriptors if descriptor["name"] in supplemental_workloads
    ]

    suite_lanes = _select_suite_lanes(
        workload_mode=workload_mode,
        primary_workload=primary_workload,
        selected_engine=selected_engine,
        num_gpus=num_gpus,
        avg_prompt_tokens=avg_prompt_tokens,
        disaggregation=disagg.get("disaggregation", {}),
        has_grace=traits.is_grace,
        has_rdma=has_rdma,
        is_nvidia=traits.is_nvidia,
    )

    overall_support = assess_benchmark_support(
        model_name=model,
        gpu_name=gpu,
        num_gpus=num_gpus,
        engine_name=selected_engine,
        workload=load_workload(primary_workload),
        prompt_tokens=avg_prompt_tokens,
        has_rdma=has_rdma,
    )
    required_unsupported_lanes: list[str] = []
    for lane in suite_lanes:
        experiment_spec = load_experiment(lane["experiment"])
        lane_support = assess_benchmark_support(
            model_name=model,
            gpu_name=gpu,
            num_gpus=max(num_gpus, 2) if lane["phase"] == "disaggregated" else num_gpus,
            engine_name=lane["engine"],
            workload=load_workload(lane["workload"]),
            experiment=experiment_spec,
            prompt_tokens=avg_prompt_tokens,
            has_rdma=has_rdma,
        )
        lane["support"] = lane_support.model_dump(mode="json")
        if lane["required"] and lane_support.status == "unsupported":
            required_unsupported_lanes.append(lane["experiment"])
        if include_stack_plans and (lane["phase"] != "disaggregated" or num_gpus >= 2):
            try:
                lane["stack_plan"] = build_benchmark_stack_plan(
                    lane["experiment"],
                    gpu,
                    num_gpus=max(num_gpus, 2) if lane["phase"] == "disaggregated" else num_gpus,
                    model=model,
                    host=host,
                ).model_dump(mode="json")
            except Exception as exc:  # noqa: BLE001
                lane["stack_plan_error"] = str(exc)
                if lane["required"] and lane["experiment"] not in required_unsupported_lanes:
                    required_unsupported_lanes.append(lane["experiment"])

    ready = overall_support.status != "unsupported" and not required_unsupported_lanes

    return {
        "benchmark_strategy": {
            "workload_mode": workload_mode.value,
            "selected_engine": selected_engine,
            "support": overall_support.model_dump(mode="json"),
            "ready": ready,
            "required_unsupported_lanes": required_unsupported_lanes,
            "recommendation": recommendation,
            "primary_workload": primary_descriptor,
            "supplemental_workloads": supplemental_descriptors,
            "matrix": matrix,
            "suite": suite_lanes,
            "kv_strategy": kv_strategy.get("strategy", {}),
            "kv_budget": kv_strategy.get("kv_budget", {}),
            "disaggregation": disagg.get("disaggregation", {}),
            "recommendation_summary": recommendation.get("summary", ""),
            "engine_alignment": {
                "selected_engine": selected_engine,
                "suite_engines": sorted({lane["engine"] for lane in suite_lanes}),
                "summary": (
                    "Packaged benchmark lanes can span different engines; the suite is curated "
                    "for operator coverage, not forced to match a single optimizer pick."
                ),
            },
            "rationale": [
                f"Use our packaged benchmark lanes as the source of truth for {workload_mode.value} optimization.",
                (
                    f"Selected engine {selected_engine} on {gpu_profile.name} "
                    f"using model class {model_variant.model_class.value}."
                ),
                (f"GPU ISA is {gpu_profile.compute_capability}; support gating is benchmark-lane specific."),
                f"Primary focus area for this scenario is {focus_area}.",
            ],
        },
        "summary": (
            f"Planned {len(suite_lanes)} benchmark lane(s) for {model_variant.name} on "
            f"{gpu_profile.name} ({workload_mode.value})"
        ),
        "confidence": 0.88 if ready else 0.72,
        "evidence": "benchmark_strategy_planner",
    }


async def plan_benchmark_strategy_with_runtime(
    model: str,
    gpu: str,
    *,
    workload: str = "chat",
    num_gpus: int = 1,
    engine: str = "auto",
    max_context: int = 32_768,
    concurrent_sessions: int = 100,
    avg_prompt_tokens: int = 4_096,
    request_rate_per_sec: float = 10.0,
    has_rdma: bool = False,
    multi_node: bool = False,
    host: str = "127.0.0.1",
    endpoint: str = "",
    include_stack_plans: bool = True,
    current_engine: str = "",
    current_model_name: str = "",
    current_model_type: str = "",
    current_attention_type: str = "",
    current_experts_total: int = 0,
    current_tp: int = 0,
    current_ep: int = 0,
    current_quantization: str = "",
    current_kv_cache_dtype: str = "",
    current_gpu_memory_utilization: float = 0.0,
    current_split_prefill_decode: bool | None = None,
    current_scheduler: dict[str, Any] | None = None,
    current_cache: dict[str, Any] | None = None,
    allow_private: bool = True,
    metrics_auth: Any = None,
    include_identity: bool = True,
) -> dict[str, Any]:
    """Plan the benchmark suite and optionally bridge it to live runtime profiling."""
    strategy = plan_benchmark_strategy(
        model,
        gpu,
        workload=workload,
        num_gpus=num_gpus,
        engine=engine,
        max_context=max_context,
        concurrent_sessions=concurrent_sessions,
        avg_prompt_tokens=avg_prompt_tokens,
        request_rate_per_sec=request_rate_per_sec,
        has_rdma=has_rdma,
        multi_node=multi_node,
        host=host,
        include_stack_plans=include_stack_plans,
    )
    if "error" in strategy:
        return strategy

    runtime_profile: dict[str, Any] | None = None
    if endpoint:
        gpu_profile = get_gpu_profile(gpu)
        runtime_profile = await profile_runtime(
            endpoint,
            engine=current_engine,
            gpu_name=gpu_profile.name if gpu_profile is not None else gpu,
            gpu_arch=gpu_profile.compute_capability if gpu_profile is not None else "",
            model_name=current_model_name,
            model_type=current_model_type,
            attention_type=current_attention_type,
            experts_total=current_experts_total,
            tp=max(current_tp, 1),
            ep=max(current_ep, 0),
            quantization=current_quantization,
            kv_cache_dtype=current_kv_cache_dtype,
            gpu_memory_utilization=current_gpu_memory_utilization,
            has_rdma=has_rdma,
            split_prefill_decode=bool(current_split_prefill_decode),
            current_scheduler=current_scheduler,
            current_cache=current_cache,
            allow_private=allow_private,
            metrics_auth=metrics_auth,
            include_identity=include_identity,
            include_tuning_preview=True,
        )

    suite = list(strategy["benchmark_strategy"]["suite"])
    suite.sort(
        key=lambda lane: _lane_priority(
            lane,
            runtime_profile=runtime_profile,
            disaggregation=strategy["benchmark_strategy"]["disaggregation"],
        )
    )
    strategy["benchmark_strategy"]["suite"] = suite
    strategy["benchmark_strategy"]["next_actions"] = _next_actions(suite, runtime_profile=runtime_profile)

    if runtime_profile is not None:
        strategy["runtime_profile"] = runtime_profile
        strategy["benchmark_strategy"]["runtime_bridge"] = {
            "active": True,
            "endpoint": endpoint,
            "prioritized_phases": [lane["phase"] for lane in suite],
            "current_hints_supplied": {
                "engine": bool(current_engine),
                "model_name": bool(current_model_name),
                "tp": current_tp > 0,
                "quantization": bool(current_quantization),
                "split_prefill_decode": current_split_prefill_decode is not None,
            },
            "summary": (f"Bridged live runtime profile into benchmark prioritization for {endpoint}"),
        }
        strategy["summary"] = f"{strategy['summary']} | runtime profile attached for {endpoint}"
        strategy["confidence"] = min(0.94, float(strategy["confidence"]) + 0.02)
        strategy["evidence"] = "benchmark_strategy_runtime_bridge"
    else:
        strategy["benchmark_strategy"]["runtime_bridge"] = {
            "active": False,
            "summary": "No live endpoint provided — strategy is planning-only.",
        }

    return strategy
