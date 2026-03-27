"""Benchmark MCP tool registration for the InferScope server."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from fastmcp import FastMCP

from inferscope.benchmarks import (
    BenchmarkExecutionProfile,
    BenchmarkGoodputSLO,
    ProceduralWorkloadOptions,
    assess_benchmark_support,
    build_benchmark_matrix,
    build_benchmark_stack_plan,
    build_default_artifact_path,
    build_run_plan,
    compare_benchmark_artifacts,
    describe_builtin_experiments,
    describe_builtin_workloads,
    load_benchmark_artifact,
    load_experiment,
    materialize_benchmark_stack_plan,
    materialize_workload,
    plan_benchmark_strategy_with_runtime,
    run_openai_replay,
)
from inferscope.config import settings
from inferscope.endpoint_auth import resolve_auth_payload
from inferscope.production_target import (
    SUPPORTED_EXPERIMENTS,
    SUPPORTED_WORKLOAD_PACK,
    build_benchmark_readiness_summary,
    build_production_contract,
    validate_production_target,
)


def _resolve_artifact_path_for_mcp(path_or_name: str) -> Path:
    """Resolve an artifact path under the benchmark directory only."""
    artifact_root = settings.benchmark_dir.resolve()
    candidate = Path(path_or_name)
    if not candidate.is_absolute():
        candidate = artifact_root / candidate
    resolved = candidate.resolve()
    if artifact_root not in resolved.parents and resolved != artifact_root:
        raise ValueError(f"Artifact path must stay under {artifact_root}")
    return resolved


def _default_stack_bundle_dir(experiment: str, gpu: str, num_gpus: int, model: str = "") -> Path:
    model_part = f"-{model}" if model else ""
    raw_name = f"{experiment}-{gpu}-{num_gpus}gpus{model_part}"
    safe_name = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in raw_name)
    return settings.benchmark_dir / "stacks" / safe_name


def _production_error(errors: list[str]) -> dict[str, Any]:
    return {
        "error": "; ".join(errors),
        "summary": "❌ Unsupported production target",
        "production_target": build_production_contract(),
        "confidence": 1.0,
        "evidence": "production_target_validation",
    }


def _build_procedural_options(
    *,
    synthetic_requests: int = 0,
    synthetic_input_tokens: int = 0,
    synthetic_output_tokens: int = 0,
    synthetic_seed: int = 42,
    context_file: str = "",
) -> ProceduralWorkloadOptions | None:
    if context_file:
        raise ValueError("context_file is not supported from MCP tools; use the CLI for local context-file expansion")
    if not any((synthetic_requests, synthetic_input_tokens, synthetic_output_tokens)):
        return None
    return ProceduralWorkloadOptions(
        request_count=(synthetic_requests or None),
        input_tokens=(synthetic_input_tokens or None),
        output_tokens=(synthetic_output_tokens or None),
        seed=synthetic_seed,
    )


def _build_execution_profile(
    *,
    request_rate: float = 0.0,
    arrival_model: str = "immediate",
    arrival_shape: float = 0.0,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
) -> BenchmarkExecutionProfile:
    resolved_arrival_model: Literal["immediate", "poisson", "gamma"] = "immediate"
    if request_rate and arrival_model in {"immediate", "poisson", "gamma"}:
        resolved_arrival_model = cast(Literal["immediate", "poisson", "gamma"], arrival_model)
    return BenchmarkExecutionProfile(
        request_rate_rps=(request_rate or None),
        arrival_model=resolved_arrival_model,
        arrival_shape=(arrival_shape or None),
        warmup_requests=warmup_requests,
        goodput_slo=BenchmarkGoodputSLO.model_validate(goodput_slo or {}),
    )


def _request_context_tokens(request: Any) -> int | None:
    metadata = getattr(request, "metadata", {})
    if not isinstance(metadata, dict):
        return None
    value = metadata.get("approx_context_tokens")
    return value if isinstance(value, int) else None


def _resolve_benchmark_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = "",
    model: str = "",
    gpu: str = "",
    num_gpus: int = 0,
    engine: str = "",
    metrics_endpoint: str = "",
    concurrency: int = 0,
    metrics_target_overrides: dict[str, str] | None = None,
    prefix_caching: bool | None = None,
    request_rate: float = 0.0,
    arrival_model: str = "immediate",
    arrival_shape: float = 0.0,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
    strict_support: bool = True,
    synthetic_requests: int = 0,
    synthetic_input_tokens: int = 0,
    synthetic_output_tokens: int = 0,
    synthetic_seed: int = 42,
    context_file: str = "",
):
    try:
        procedural_options = _build_procedural_options(
            synthetic_requests=synthetic_requests,
            synthetic_input_tokens=synthetic_input_tokens,
            synthetic_output_tokens=synthetic_output_tokens,
            synthetic_seed=synthetic_seed,
            context_file=context_file,
        )
        input_workload_pack = materialize_workload(workload, options=procedural_options)
        if input_workload_pack.name != SUPPORTED_WORKLOAD_PACK:
            return (
                _production_error(["InferScope MCP exposes only the Kimi-K2.5 long-context coding workload."]),
                None,
                None,
                None,
                None,
            )
        experiment_spec = load_experiment(experiment) if experiment else None
        if experiment and experiment_spec is not None and experiment_spec.name not in SUPPORTED_EXPERIMENTS:
            return (
                _production_error(
                    ["InferScope MCP exposes only the Kimi-targeted vLLM and Dynamo benchmark experiments."]
                ),
                None,
                None,
                None,
                None,
            )
        if experiment_spec and input_workload_pack.name != experiment_spec.workload:
            return (
                {
                    "error": (
                        f"Workload '{input_workload_pack.name}' does not match experiment "
                        f"'{experiment_spec.name}' workload '{experiment_spec.workload}'"
                    ),
                    "summary": f"❌ Workload does not match experiment: {input_workload_pack.name}",
                    "confidence": 1.0,
                    "evidence": "benchmark_plan_resolution",
                },
                None,
                None,
                None,
                None,
            )

        workload_reference = experiment_spec.workload if experiment_spec else workload
        workload_pack = (
            materialize_workload(workload_reference, options=procedural_options)
            if experiment_spec
            else input_workload_pack
        )
        execution = _build_execution_profile(
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
        )
        selected_model_name = model or (experiment_spec.model if experiment_spec else None) or workload_pack.model or ""
        production_errors = validate_production_target(
            model_name=selected_model_name,
            gpu_name=gpu,
            workload=workload_pack.workload_class,
            engine=(engine or (experiment_spec.engine if experiment_spec else "")),
        )
        if production_errors:
            return (_production_error(production_errors), None, None, None, None)
        support = assess_benchmark_support(
            model_name=selected_model_name,
            gpu_name=gpu,
            num_gpus=(num_gpus or None),
            engine_name=(engine or (experiment_spec.engine if experiment_spec else "")),
            workload=workload_pack,
            experiment=experiment_spec,
            prompt_tokens=max(
                (
                    value
                    for request in workload_pack.requests
                    if (value := _request_context_tokens(request)) is not None
                ),
                default=0,
            )
            or None,
        )
        if strict_support and support.status == "unsupported":
            error_messages = [issue.message for issue in support.issues if issue.severity == "error"]
            return (
                {
                    "error": "; ".join(error_messages) or "Unsupported benchmark configuration",
                    "support": support.model_dump(mode="json"),
                    "summary": "❌ Unsupported benchmark configuration",
                    "confidence": 1.0,
                    "evidence": "benchmark_support_assessment",
                },
                None,
                None,
                None,
                None,
            )
        run_plan = build_run_plan(
            workload_pack,
            endpoint,
            workload_ref=workload_reference,
            experiment=experiment_spec,
            model=(model or None),
            concurrency=(concurrency or None),
            metrics_endpoint=(metrics_endpoint or None),
            metrics_target_overrides=metrics_target_overrides or {},
            prefix_caching=prefix_caching,
            execution=execution,
            support=support,
        )
    except Exception as exc:  # noqa: BLE001
        return (
            {
                "error": str(exc),
                "summary": "❌ Failed to resolve benchmark plan",
                "confidence": 1.0,
                "evidence": "benchmark_plan_resolution",
            },
            None,
            None,
            None,
            None,
        )

    return None, workload_reference, workload_pack, run_plan, run_plan.support


def register_benchmark_tools(mcp: FastMCP) -> None:
    """Register evaluation and benchmark MCP tools."""

    @mcp.tool()
    async def tool_get_production_contract() -> dict[str, Any]:
        """Return the supported Kimi production contract for MCP clients."""
        contract = build_production_contract()
        return {
            "summary": (
                "InferScope MCP is currently scoped to Kimi-K2.5 on Hopper/Blackwell, "
                "with Dynamo as the serving target and vLLM plus Dynamo as benchmark engines."
            ),
            "production_target": contract,
            "confidence": 1.0,
            "evidence": "production_target_contract",
        }

    @mcp.tool()
    async def tool_list_benchmark_workloads() -> dict[str, Any]:
        """List the production workload pack exposed by the narrowed MCP surface."""
        workloads = [SUPPORTED_WORKLOAD_PACK]
        descriptors = [
            descriptor for descriptor in describe_builtin_workloads() if descriptor["name"] == SUPPORTED_WORKLOAD_PACK
        ]
        return {
            "summary": f"{len(workloads)} built-in workload pack(s) available",
            "workloads": workloads,
            "descriptors": descriptors,
            "procedural_workloads": [SUPPORTED_WORKLOAD_PACK],
            "production_target": build_production_contract(),
            "confidence": 1.0,
            "evidence": "packaged_workload_catalog",
        }

    @mcp.tool()
    async def tool_list_benchmark_experiments() -> dict[str, Any]:
        """List the Kimi-targeted vLLM and Dynamo experiments exposed by MCP."""
        experiments = list(SUPPORTED_EXPERIMENTS)
        descriptors = [
            descriptor for descriptor in describe_builtin_experiments() if descriptor["name"] in SUPPORTED_EXPERIMENTS
        ]
        return {
            "summary": f"{len(experiments)} built-in benchmark experiment(s) available",
            "experiments": experiments,
            "descriptors": descriptors,
            "production_target": build_production_contract(),
            "confidence": 1.0,
            "evidence": "packaged_experiment_catalog",
        }

    @mcp.tool()
    async def tool_get_benchmark_matrix(
        gpu_family: str = "",
        model_class: str = "",
        workload_class: str = "",
        focus_area: str = "",
        engine: str = "",
    ) -> dict[str, Any]:
        """Return the structured benchmark matrix filtered by GPU/model/workload intent."""
        if engine and engine.strip().lower() not in {"", "auto", "dynamo", "vllm"}:
            return {
                "summary": "0 workload(s), 0 experiment(s), 0 suggested pairing(s)",
                "matrix": {
                    "filters": {"engine": engine},
                    "workloads": [],
                    "experiments": [],
                    "suggested_pairs": [],
                },
                "production_target": build_production_contract(),
                "confidence": 1.0,
                "evidence": "benchmark_matrix_catalog",
            }
        matrix = build_benchmark_matrix(
            gpu_family=gpu_family,
            model_class=model_class,
            workload_class=workload_class,
            focus_area=focus_area,
            engine=engine,
        )
        matrix["workloads"] = [
            descriptor for descriptor in matrix["workloads"] if descriptor["name"] == SUPPORTED_WORKLOAD_PACK
        ]
        matrix["experiments"] = [
            descriptor for descriptor in matrix["experiments"] if descriptor["name"] in SUPPORTED_EXPERIMENTS
        ]
        matrix["suggested_pairs"] = [
            pair for pair in matrix["suggested_pairs"] if pair["experiment"] in SUPPORTED_EXPERIMENTS
        ]
        return {
            "summary": (
                f"{len(matrix['workloads'])} workload(s), {len(matrix['experiments'])} experiment(s), "
                f"{len(matrix['suggested_pairs'])} suggested pairing(s)"
            ),
            "matrix": matrix,
            "production_target": build_production_contract(),
            "confidence": 1.0,
            "evidence": "benchmark_matrix_catalog",
        }

    @mcp.tool()
    async def tool_plan_benchmark_strategy(
        model: str,
        gpu: str,
        workload: str = "chat",
        num_gpus: int = 1,
        engine: str = "auto",
        max_context: int = 32768,
        concurrent_sessions: int = 100,
        avg_prompt_tokens: int = 4096,
        request_rate_per_sec: float = 10.0,
        has_rdma: bool = False,
        host: str = "127.0.0.1",
        endpoint: str = "",
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
        provider: str = "",
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Plan the right benchmark suite and optionally bridge it to a live runtime profile."""
        result = await plan_benchmark_strategy_with_runtime(
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
            host=host,
            endpoint=endpoint,
            current_engine=current_engine,
            current_model_name=current_model_name,
            current_model_type=current_model_type,
            current_attention_type=current_attention_type,
            current_experts_total=current_experts_total,
            current_tp=current_tp,
            current_ep=current_ep,
            current_quantization=current_quantization,
            current_kv_cache_dtype=current_kv_cache_dtype,
            current_gpu_memory_utilization=current_gpu_memory_utilization,
            current_split_prefill_decode=current_split_prefill_decode,
            current_scheduler=current_scheduler,
            current_cache=current_cache,
            allow_private=False,
            metrics_auth=resolve_auth_payload(metrics_auth, provider=provider),
            include_identity=True,
        )
        result["confidence"] = min(0.96, float(result.get("confidence", 0.85)))
        result["evidence"] = "benchmark_strategy_runtime_bridge" if endpoint else "benchmark_strategy_planner"
        return result

    @mcp.tool()
    async def tool_generate_benchmark_stack_plan(
        experiment: str,
        gpu: str,
        num_gpus: int = 2,
        model: str = "",
        prefix_caching: bool | None = None,
        host: str = "127.0.0.1",
        enable_trace: bool = False,
        otlp_endpoint: str = "",
        vllm_proxy_command: str = "",
    ) -> dict[str, Any]:
        """Generate live launch commands for the supported Kimi benchmark stacks."""
        if experiment not in SUPPORTED_EXPERIMENTS:
            return {
                "error": f"Unknown built-in experiment '{experiment}'",
                "available_experiments": list(SUPPORTED_EXPERIMENTS),
                "summary": f"❌ Unknown built-in experiment: {experiment}",
                "production_target": build_production_contract(),
                "confidence": 1.0,
                "evidence": "builtin_experiment_catalog",
            }
        experiment_spec = load_experiment(experiment)
        production_errors = validate_production_target(
            model_name=(model or "Kimi-K2.5"),
            gpu_name=gpu,
            workload="coding",
            engine=experiment_spec.engine,
            num_gpus=num_gpus,
            topology_mode=experiment_spec.topology.mode,
        )
        if production_errors:
            return _production_error(production_errors)
        try:
            stack_plan = build_benchmark_stack_plan(
                experiment,
                gpu,
                num_gpus,
                model=model,
                host=host,
                enable_trace=enable_trace,
                otlp_endpoint=otlp_endpoint,
                request_id_headers=["x-request-id", "x-trace-id"],
                vllm_proxy_command=vllm_proxy_command,
            )
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "summary": "❌ Failed to generate benchmark stack plan",
                "confidence": 1.0,
                "evidence": "benchmark_stack_plan",
            }
        return {
            "summary": f"Generated launch plan for {stack_plan.experiment}",
            "stack_plan": stack_plan.model_dump(mode="json"),
            "support": stack_plan.support,
            "benchmark_command": stack_plan.benchmark_command,
            "production_target": build_production_contract(),
            "confidence": 0.9,
            "evidence": "benchmark_stack_plan",
        }

    @mcp.tool()
    async def tool_materialize_benchmark_stack_plan(
        experiment: str,
        gpu: str,
        num_gpus: int = 2,
        model: str = "",
        prefix_caching: bool | None = None,
        host: str = "127.0.0.1",
        enable_trace: bool = False,
        otlp_endpoint: str = "",
        vllm_proxy_command: str = "",
        output_dir: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Write a runnable benchmark stack bundle under the benchmark artifact directory."""
        if experiment not in SUPPORTED_EXPERIMENTS:
            return {
                "error": f"Unknown built-in experiment '{experiment}'",
                "available_experiments": list(SUPPORTED_EXPERIMENTS),
                "summary": f"❌ Unknown built-in experiment: {experiment}",
                "production_target": build_production_contract(),
                "confidence": 1.0,
                "evidence": "builtin_experiment_catalog",
            }
        experiment_spec = load_experiment(experiment)
        production_errors = validate_production_target(
            model_name=(model or "Kimi-K2.5"),
            gpu_name=gpu,
            workload="coding",
            engine=experiment_spec.engine,
            num_gpus=num_gpus,
            topology_mode=experiment_spec.topology.mode,
        )
        if production_errors:
            return _production_error(production_errors)
        try:
            stack_plan = build_benchmark_stack_plan(
                experiment,
                gpu,
                num_gpus,
                model=model,
                host=host,
                enable_trace=enable_trace,
                otlp_endpoint=otlp_endpoint,
                request_id_headers=["x-request-id", "x-trace-id"],
                vllm_proxy_command=vllm_proxy_command,
            )
            bundle_dir = (
                _resolve_artifact_path_for_mcp(output_dir)
                if output_dir
                else _default_stack_bundle_dir(experiment, gpu, num_gpus, model).resolve()
            )
            materialized = materialize_benchmark_stack_plan(stack_plan, bundle_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "summary": "❌ Failed to materialize benchmark stack plan",
                "confidence": 1.0,
                "evidence": "benchmark_stack_materialization",
            }
        return {
            "summary": f"Materialized runnable stack for {stack_plan.experiment}",
            "materialized": materialized.model_dump(mode="json"),
            "support": stack_plan.support,
            "benchmark_command": stack_plan.benchmark_command,
            "production_target": build_production_contract(),
            "confidence": 0.95,
            "evidence": "benchmark_stack_materialization",
        }

    @mcp.tool()
    async def tool_resolve_benchmark_plan(
        workload: str,
        endpoint: str,
        experiment: str = "",
        model: str = "",
        gpu: str = "",
        num_gpus: int = 0,
        engine: str = "",
        metrics_endpoint: str = "",
        concurrency: int = 0,
        metrics_target_overrides: dict[str, str] | None = None,
        prefix_caching: bool | None = None,
        request_rate: float = 0.0,
        arrival_model: str = "immediate",
        arrival_shape: float = 0.0,
        warmup_requests: int = 0,
        goodput_slo: dict[str, Any] | None = None,
        strict_support: bool = True,
        synthetic_requests: int = 0,
        synthetic_input_tokens: int = 0,
        synthetic_output_tokens: int = 0,
        synthetic_seed: int = 42,
        context_file: str = "",
    ) -> dict[str, Any]:
        """Resolve a workload reference and optional experiment reference into a concrete run plan."""
        error, workload_reference, _, run_plan, support = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            model=model,
            gpu=gpu,
            num_gpus=num_gpus,
            engine=engine,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
            prefix_caching=prefix_caching,
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
            strict_support=strict_support,
            synthetic_requests=synthetic_requests,
            synthetic_input_tokens=synthetic_input_tokens,
            synthetic_output_tokens=synthetic_output_tokens,
            synthetic_seed=synthetic_seed,
            context_file=context_file,
        )
        if error is not None:
            return cast(dict[str, Any], error)
        return {
            "summary": f"Resolved benchmark plan for {workload_reference}",
            "run_plan": cast(dict[str, Any], run_plan.model_dump(mode="json")),
            "support": cast(dict[str, Any], support.model_dump(mode="json")) if support is not None else None,
            "production_target": build_production_contract(),
            "confidence": 0.95,
            "evidence": "benchmark_plan_resolution",
        }

    @mcp.tool()
    async def tool_run_benchmark(
        workload: str,
        endpoint: str,
        experiment: str = "",
        model: str = "",
        gpu: str = "",
        num_gpus: int = 0,
        engine: str = "",
        metrics_endpoint: str = "",
        concurrency: int = 0,
        capture_metrics: bool = True,
        save_artifact: bool = True,
        metrics_target_overrides: dict[str, str] | None = None,
        prefix_caching: bool | None = None,
        request_rate: float = 0.0,
        arrival_model: str = "immediate",
        arrival_shape: float = 0.0,
        warmup_requests: int = 0,
        goodput_slo: dict[str, Any] | None = None,
        strict_support: bool = True,
        synthetic_requests: int = 0,
        synthetic_input_tokens: int = 0,
        synthetic_output_tokens: int = 0,
        synthetic_seed: int = 42,
        context_file: str = "",
        provider: str = "",
        metrics_provider: str = "",
        request_auth: dict | None = None,
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Replay a workload reference against an OpenAI-compatible endpoint."""
        error, workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            model=model,
            gpu=gpu,
            num_gpus=num_gpus,
            engine=engine,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
            prefix_caching=prefix_caching,
            request_rate=request_rate,
            arrival_model=arrival_model,
            arrival_shape=arrival_shape,
            warmup_requests=warmup_requests,
            goodput_slo=goodput_slo,
            strict_support=strict_support,
            synthetic_requests=synthetic_requests,
            synthetic_input_tokens=synthetic_input_tokens,
            synthetic_output_tokens=synthetic_output_tokens,
            synthetic_seed=synthetic_seed,
            context_file=context_file,
        )
        if error is not None:
            return cast(dict[str, Any], error)

        try:
            request_auth_config = resolve_auth_payload(request_auth, provider=provider)
            metrics_auth_config = resolve_auth_payload(
                metrics_auth,
                provider=metrics_provider or provider,
            )
            artifact = await run_openai_replay(
                workload_pack,
                endpoint,
                metrics_endpoint=(metrics_endpoint or None),
                run_plan=run_plan,
                workload_ref=workload_reference,
                provider=provider,
                metrics_provider=metrics_provider,
                api_key=(request_auth_config.api_key or None) if request_auth_config else None,
                auth_scheme=request_auth_config.auth_scheme if request_auth_config else "",
                auth_header_name=request_auth_config.auth_header_name if request_auth_config else "",
                extra_headers=request_auth_config.headers if request_auth_config else None,
                metrics_api_key=(metrics_auth_config.api_key or None) if metrics_auth_config else None,
                metrics_auth_scheme=metrics_auth_config.auth_scheme if metrics_auth_config else "",
                metrics_auth_header_name=(metrics_auth_config.auth_header_name if metrics_auth_config else ""),
                metrics_headers=metrics_auth_config.headers if metrics_auth_config else None,
                capture_metrics=capture_metrics,
                allow_private=False,
            )
            artifact_path = ""
            if save_artifact:
                artifact_path = str(artifact.save_json(build_default_artifact_path(artifact)))
        except Exception as exc:  # noqa: BLE001
            return {
                "error": str(exc),
                "summary": "❌ Benchmark run failed",
                "confidence": 1.0,
                "evidence": "live_benchmark_replay",
            }
        return {
            "summary": (
                f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                "requests succeeded"
            ),
            "artifact_path": artifact_path,
            "benchmark_id": artifact.benchmark_id,
            "run_plan": cast(dict[str, Any], run_plan.model_dump(mode="json")),
            "support": cast(dict[str, Any], support.model_dump(mode="json")) if support is not None else None,
            "observed_runtime": (
                cast(dict[str, Any], artifact.run_plan.get("observed_runtime", {})) if artifact.run_plan else {}
            ),
            "benchmark_summary": cast(dict[str, Any], artifact.summary.model_dump(mode="json")),
            "production_readiness": build_benchmark_readiness_summary(artifact),
            "production_target": build_production_contract(),
            "confidence": 0.85,
            "evidence": "live_benchmark_replay",
        }

    @mcp.tool()
    async def tool_compare_benchmarks(baseline_artifact: str, candidate_artifact: str) -> dict[str, Any]:
        """Compare two saved benchmark artifacts and report latency/TTFT deltas."""
        baseline = load_benchmark_artifact(_resolve_artifact_path_for_mcp(baseline_artifact))
        candidate = load_benchmark_artifact(_resolve_artifact_path_for_mcp(candidate_artifact))
        comparison = compare_benchmark_artifacts(baseline, candidate)
        comparison["confidence"] = 0.9
        comparison["evidence"] = "benchmark_artifact_comparison"
        return comparison

    @mcp.tool()
    async def tool_get_benchmark_artifact(artifact_name: str) -> dict[str, Any]:
        """Read a saved benchmark artifact by filename from the benchmark directory."""
        artifact = load_benchmark_artifact(_resolve_artifact_path_for_mcp(artifact_name))
        return {
            "summary": f"Loaded benchmark artifact {artifact.default_filename}",
            "artifact": cast(dict[str, Any], artifact.model_dump(mode="json")),
            "observed_runtime": (
                cast(dict[str, Any], artifact.run_plan.get("observed_runtime", {})) if artifact.run_plan else {}
            ),
            "production_readiness": build_benchmark_readiness_summary(artifact),
            "production_target": build_production_contract(),
            "confidence": 1.0,
            "evidence": "saved_benchmark_artifact",
        }
