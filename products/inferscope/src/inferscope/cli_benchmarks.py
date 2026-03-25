"""Benchmark command registration for the InferScope CLI."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

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
    list_builtin_experiments,
    list_builtin_workloads,
    load_benchmark_artifact,
    load_experiment,
    materialize_benchmark_stack_plan,
    materialize_workload,
    parse_metrics_target_overrides,
    plan_benchmark_strategy_with_runtime,
    run_openai_replay,
)
from inferscope.endpoint_auth import parse_header_values


def _parse_json_option(raw: str, *, option_name: str) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise typer.BadParameter(f"{option_name} must be valid JSON") from exc
    if not isinstance(value, dict):
        raise typer.BadParameter(f"{option_name} must be a JSON object")
    return {str(key): val for key, val in value.items()}


def _build_procedural_options(
    *,
    synthetic_requests: int | None = None,
    synthetic_input_tokens: int | None = None,
    synthetic_output_tokens: int | None = None,
    synthetic_seed: int = 42,
    context_file: str = "",
) -> ProceduralWorkloadOptions | None:
    if not any(
        value is not None and value != ""
        for value in (
            synthetic_requests,
            synthetic_input_tokens,
            synthetic_output_tokens,
            context_file,
        )
    ):
        return None
    return ProceduralWorkloadOptions(
        request_count=synthetic_requests,
        input_tokens=synthetic_input_tokens,
        output_tokens=synthetic_output_tokens,
        seed=synthetic_seed,
        context_file=(context_file or None),
    )


def _build_execution_profile(
    *,
    request_rate: float | None = None,
    arrival_model: str = "immediate",
    arrival_shape: float | None = None,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
) -> BenchmarkExecutionProfile:
    return BenchmarkExecutionProfile(
        request_rate_rps=request_rate,
        arrival_model=("immediate" if request_rate in (None, 0.0) else arrival_model),
        arrival_shape=arrival_shape,
        warmup_requests=warmup_requests,
        goodput_slo=BenchmarkGoodputSLO.model_validate(goodput_slo or {}),
    )


def _resolve_benchmark_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = "",
    model: str = "",
    gpu: str = "",
    num_gpus: int | None = None,
    engine: str = "",
    concurrency: int | None = None,
    metrics_endpoint: str | None = None,
    metrics_target: list[str] | None = None,
    topology_mode: str = "",
    session_routing: str = "",
    session_header_name: str = "",
    cache_strategy: str = "",
    cache_tier: list[str] | None = None,
    cache_connector: str = "",
    session_affinity: bool | None = None,
    request_rate: float | None = None,
    arrival_model: str = "immediate",
    arrival_shape: float | None = None,
    warmup_requests: int = 0,
    goodput_slo: dict[str, Any] | None = None,
    strict_support: bool = True,
    synthetic_requests: int | None = None,
    synthetic_input_tokens: int | None = None,
    synthetic_output_tokens: int | None = None,
    synthetic_seed: int = 42,
    context_file: str = "",
):
    procedural_options = _build_procedural_options(
        synthetic_requests=synthetic_requests,
        synthetic_input_tokens=synthetic_input_tokens,
        synthetic_output_tokens=synthetic_output_tokens,
        synthetic_seed=synthetic_seed,
        context_file=context_file,
    )
    input_workload_pack = materialize_workload(workload, options=procedural_options)
    experiment_spec = load_experiment(experiment) if experiment else None
    if experiment_spec and input_workload_pack.name != experiment_spec.workload:
        raise ValueError(
            f"Workload '{input_workload_pack.name}' does not match experiment "
            f"'{experiment_spec.name}' workload '{experiment_spec.workload}'"
        )

    workload_reference = experiment_spec.workload if experiment_spec else workload
    workload_pack = (
        materialize_workload(workload_reference, options=procedural_options) if experiment_spec else input_workload_pack
    )
    metrics_target_overrides = parse_metrics_target_overrides(metrics_target)
    execution = _build_execution_profile(
        request_rate=request_rate,
        arrival_model=arrival_model,
        arrival_shape=arrival_shape,
        warmup_requests=warmup_requests,
        goodput_slo=goodput_slo,
    )
    support = assess_benchmark_support(
        model_name=(model or (experiment_spec.model if experiment_spec else None) or workload_pack.model or ""),
        gpu_name=gpu,
        num_gpus=num_gpus,
        engine_name=(engine or (experiment_spec.engine if experiment_spec else "")),
        workload=workload_pack,
        experiment=experiment_spec,
        prompt_tokens=max(
            (
                int(request.metadata.get("approx_context_tokens"))
                for request in workload_pack.requests
                if isinstance(request.metadata.get("approx_context_tokens"), int)
            ),
            default=0,
        )
        or None,
    )
    if strict_support and support.status == "unsupported":
        messages = [issue.message for issue in support.issues if issue.severity == "error"]
        raise ValueError("; ".join(messages) or "Unsupported benchmark configuration")
    run_plan = build_run_plan(
        workload_pack,
        endpoint,
        workload_ref=workload_reference,
        experiment=experiment_spec,
        model=(model or None),
        concurrency=concurrency,
        metrics_endpoint=metrics_endpoint,
        metrics_target_overrides=metrics_target_overrides,
        topology_mode=(topology_mode or None),
        session_routing=(session_routing or None),
        session_header_name=(session_header_name or None),
        cache_strategy=(cache_strategy or None),
        cache_tiers=(cache_tier or None),
        cache_connector=(cache_connector or None),
        session_affinity=session_affinity,
        execution=execution,
        support=support,
    )
    return workload_reference, workload_pack, run_plan, support


def register_benchmark_commands(
    app: typer.Typer,
    *,
    print_result: Callable[[dict[str, Any]], None],
    resolve_metrics_auth: Callable[..., Any] | None = None,
) -> None:
    """Register benchmark and evaluation commands on the main CLI app."""

    @app.command(name="benchmark-workloads")
    def benchmark_workloads_cmd():
        """List built-in benchmark workload packs."""
        workloads = list_builtin_workloads()
        print_result(
            {
                "summary": f"{len(workloads)} built-in workload pack(s) available",
                "workloads": workloads,
                "descriptors": describe_builtin_workloads(),
                "procedural_workloads": ["tool-agent", "coding-long-context"],
            }
        )

    @app.command(name="benchmark-experiments")
    def benchmark_experiments_cmd():
        """List built-in disaggregation/cache benchmark experiment specs."""
        experiments = list_builtin_experiments()
        print_result(
            {
                "summary": f"{len(experiments)} built-in benchmark experiment(s) available",
                "experiments": experiments,
                "descriptors": describe_builtin_experiments(),
            }
        )

    @app.command(name="benchmark-matrix")
    def benchmark_matrix_cmd(
        gpu_family: Annotated[str, typer.Option(help="Filter by target GPU family")] = "",
        model_class: Annotated[str, typer.Option(help="Filter by target model class")] = "",
        workload_class: Annotated[str, typer.Option(help="Filter by workload class")] = "",
        focus_area: Annotated[str, typer.Option(help="Filter by focus area")] = "",
        engine: Annotated[str, typer.Option(help="Filter by engine")] = "",
    ):
        """Show the benchmark matrix organized by GPU/model/workload intent."""
        matrix = build_benchmark_matrix(
            gpu_family=gpu_family,
            model_class=model_class,
            workload_class=workload_class,
            focus_area=focus_area,
            engine=engine,
        )
        print_result(
            {
                "summary": (
                    f"{len(matrix['workloads'])} workload(s), {len(matrix['experiments'])} experiment(s), "
                    f"{len(matrix['suggested_pairs'])} suggested pairing(s)"
                ),
                "matrix": matrix,
            }
        )

    @app.command(name="benchmark-strategy")
    def benchmark_strategy_cmd(
        model: Annotated[str, typer.Argument(help="Model name")],
        gpu: Annotated[str, typer.Argument(help="GPU type")],
        workload: Annotated[str, typer.Option(help="Workload: coding, chat, agent, long_context_rag")] = "chat",
        num_gpus: Annotated[int, typer.Option(help="Number of GPUs", min=1)] = 1,
        engine: Annotated[str, typer.Option(help="Engine target or auto")] = "auto",
        max_context: Annotated[int, typer.Option(help="Target max context length", min=1024)] = 32768,
        concurrent_sessions: Annotated[int, typer.Option(help="Concurrent sessions to model", min=1)] = 100,
        avg_prompt_tokens: Annotated[int, typer.Option(help="Average prompt tokens", min=1)] = 4096,
        request_rate_per_sec: Annotated[float, typer.Option(help="Requests per second", min=0.0)] = 10.0,
        has_rdma: Annotated[bool, typer.Option(help="RDMA available between nodes")] = False,
        host: Annotated[str, typer.Option(help="Host/IP for generated stack plans")] = "127.0.0.1",
        endpoint: Annotated[str, typer.Option(help="Optional live endpoint to profile and bridge into the plan")] = "",
        current_engine: Annotated[
            str, typer.Option(help="Current live engine if different from the planned target")
        ] = "",
        current_model_name: Annotated[str, typer.Option(help="Current live model name if known")] = "",
        current_model_type: Annotated[str, typer.Option(help="Current live model type: dense or moe")] = "",
        current_attention_type: Annotated[
            str, typer.Option(help="Current live attention type: GQA, MLA, MHA, hybrid")
        ] = "",
        current_experts_total: Annotated[int, typer.Option(help="Current live expert count for MoE models", min=0)] = 0,
        current_tp: Annotated[int, typer.Option(help="Current live tensor parallelism degree", min=0)] = 0,
        current_ep: Annotated[int, typer.Option(help="Current live expert parallelism degree", min=0)] = 0,
        current_quantization: Annotated[str, typer.Option(help="Current live quantization")] = "",
        current_kv_cache_dtype: Annotated[str, typer.Option(help="Current live KV cache dtype")] = "",
        current_gpu_memory_utilization: Annotated[
            float,
            typer.Option(help="Current live GPU memory utilization if known", min=0.0, max=1.0),
        ] = 0.0,
        current_split_prefill_decode: Annotated[
            bool | None,
            typer.Option(help="Current live deployment already uses split prefill/decode"),
        ] = None,
        current_scheduler: Annotated[
            str,
            typer.Option(help="Optional scheduler JSON object for runtime bridge"),
        ] = "",
        current_cache: Annotated[
            str,
            typer.Option(help="Optional cache JSON object for runtime bridge"),
        ] = "",
        provider: Annotated[
            str,
            typer.Option(help="Managed provider preset for authenticated metrics endpoints"),
        ] = "",
        metrics_api_key: Annotated[
            str,
            typer.Option(help="API key for scraping authenticated metrics endpoints"),
        ] = "",
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help="Metrics auth scheme: bearer, api-key, x-api-key, raw"),
        ] = "",
        metrics_auth_header_name: Annotated[
            str,
            typer.Option(help="Override metrics auth header name"),
        ] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
    ):
        """Plan the benchmark suite and optionally bridge it to a live runtime profile."""
        result = asyncio.run(
            plan_benchmark_strategy_with_runtime(
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
                current_scheduler=_parse_json_option(current_scheduler, option_name="current scheduler"),
                current_cache=_parse_json_option(current_cache, option_name="current cache"),
                allow_private=True,
                metrics_auth=(
                    resolve_metrics_auth(
                        provider=provider,
                        metrics_api_key=metrics_api_key,
                        metrics_auth_scheme=metrics_auth_scheme,
                        metrics_auth_header_name=metrics_auth_header_name,
                        metrics_header=metrics_header,
                    )
                    if resolve_metrics_auth is not None and endpoint
                    else None
                ),
            )
        )
        print_result(result)

    @app.command(name="benchmark-stack-plan")
    def benchmark_stack_plan_cmd(
        experiment: Annotated[str, typer.Argument(help="Built-in experiment spec name")],
        gpu: Annotated[str, typer.Argument(help="GPU type for the planned stack")],
        num_gpus: Annotated[int, typer.Option(help="Total GPU count available to the stack", min=1)] = 2,
        model: Annotated[str, typer.Option(help="Override model name from the experiment/workload")] = "",
        host: Annotated[str, typer.Option(help="Host/IP used in generated endpoints and commands")] = "127.0.0.1",
        enable_trace: Annotated[bool, typer.Option(help="Enable router OTLP tracing when supported")] = False,
        otlp_endpoint: Annotated[str, typer.Option(help="OTLP traces endpoint for router tracing")] = "",
        vllm_proxy_command: Annotated[
            str,
            typer.Option(help="Concrete proxy command for vLLM disaggregated-prefill bundles"),
        ] = "",
    ):
        """Generate launch commands and benchmark wiring for a live vLLM/SGLang experiment stack."""
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
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                "summary": f"Generated launch plan for {stack_plan.experiment}",
                "stack_plan": stack_plan.model_dump(mode="json"),
                "support": stack_plan.support,
                "benchmark_command": stack_plan.benchmark_command,
            }
        )

    @app.command(name="benchmark-stack-write")
    def benchmark_stack_write_cmd(
        experiment: Annotated[str, typer.Argument(help="Built-in experiment spec name")],
        gpu: Annotated[str, typer.Argument(help="GPU type for the planned stack")],
        output_dir: Annotated[Path, typer.Argument(help="Directory to write scripts and configs into")],
        num_gpus: Annotated[int, typer.Option(help="Total GPU count available to the stack", min=1)] = 2,
        model: Annotated[str, typer.Option(help="Override model name from the experiment/workload")] = "",
        host: Annotated[str, typer.Option(help="Host/IP used in generated endpoints and commands")] = "127.0.0.1",
        enable_trace: Annotated[bool, typer.Option(help="Enable router OTLP tracing when supported")] = False,
        otlp_endpoint: Annotated[str, typer.Option(help="OTLP traces endpoint for router tracing")] = "",
        vllm_proxy_command: Annotated[
            str,
            typer.Option(help="Concrete proxy command for vLLM disaggregated-prefill bundles"),
        ] = "",
        overwrite: Annotated[bool, typer.Option(help="Allow writing into a non-empty output directory")] = False,
    ):
        """Write a runnable benchmark stack bundle with scripts, config files, and plan metadata."""
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
            materialized = materialize_benchmark_stack_plan(stack_plan, output_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                "summary": f"Materialized runnable stack for {stack_plan.experiment}",
                "materialized": materialized.model_dump(mode="json"),
                "support": stack_plan.support,
                "benchmark_command": stack_plan.benchmark_command,
            }
        )

    @app.command(name="benchmark-plan")
    def benchmark_plan_cmd(
        workload: Annotated[str, typer.Argument(help="Workload file path or built-in workload name")],
        endpoint: Annotated[str, typer.Argument(help="OpenAI-compatible request endpoint base URL")],
        experiment: Annotated[str, typer.Option(help="Optional built-in or file-backed experiment spec")] = "",
        model: Annotated[str, typer.Option(help="Override model name from the workload/experiment")] = "",
        gpu: Annotated[str, typer.Option(help="Concrete GPU SKU for support validation")] = "",
        num_gpus: Annotated[int | None, typer.Option(help="Concrete GPU count for support validation", min=1)] = None,
        engine: Annotated[str, typer.Option(help="Override engine for support validation")] = "",
        concurrency: Annotated[int | None, typer.Option(help="Override concurrency", min=1)] = None,
        metrics_endpoint: Annotated[str | None, typer.Option(help="Optional default Prometheus base URL")] = None,
        metrics_target: Annotated[
            list[str] | None,
            typer.Option(
                help="Additional metrics targets as name=url. Example: --metrics-target router=http://host:9000"
            ),
        ] = None,
        topology_mode: Annotated[str, typer.Option(help="Override topology mode")] = "",
        session_routing: Annotated[str, typer.Option(help="Override session routing mode")] = "",
        session_header_name: Annotated[str, typer.Option(help="Override session header name")] = "",
        cache_strategy: Annotated[str, typer.Option(help="Override cache strategy")] = "",
        cache_tier: Annotated[list[str] | None, typer.Option(help="Override cache tiers")] = None,
        cache_connector: Annotated[str, typer.Option(help="Override cache connector name")] = "",
        session_affinity: Annotated[bool | None, typer.Option(help="Override session affinity")] = None,
        request_rate: Annotated[float | None, typer.Option(help="Request rate for scheduled replay", min=0.0)] = None,
        arrival_model: Annotated[str, typer.Option(help="Arrival model: immediate, poisson, gamma")] = "immediate",
        arrival_shape: Annotated[float | None, typer.Option(help="Gamma arrival shape", min=0.0001)] = None,
        warmup_requests: Annotated[int, typer.Option(help="Warmup requests before measurement", min=0)] = 0,
        goodput_slo: Annotated[
            str,
            typer.Option(help="Optional JSON object for goodput thresholds"),
        ] = "",
        strict_support: Annotated[
            bool,
            typer.Option(
                "--strict-support/--no-strict-support",
                help="Fail plan resolution on unsupported GPU/model/engine combinations",
            ),
        ] = True,
        synthetic_requests: Annotated[
            int | None,
            typer.Option(
                help="Procedurally expand the workload to this many requests",
                min=1,
            ),
        ] = None,
        synthetic_input_tokens: Annotated[
            int | None,
            typer.Option(
                help="Approximate input token target for procedural workloads",
                min=64,
            ),
        ] = None,
        synthetic_output_tokens: Annotated[
            int | None,
            typer.Option(
                help="Approximate output token target for procedural workloads",
                min=32,
            ),
        ] = None,
        synthetic_seed: Annotated[
            int,
            typer.Option(help="Seed for procedural workload expansion", min=0),
        ] = 42,
        context_file: Annotated[
            str,
            typer.Option(help=("Optional repo/context file used when procedurally expanding coding-long-context")),
        ] = "",
    ):
        """Resolve workload + experiment + runtime overrides into a concrete benchmark run plan."""
        try:
            workload_reference, _, run_plan, support = _resolve_benchmark_plan(
                workload,
                endpoint,
                experiment=experiment,
                model=model,
                gpu=gpu,
                num_gpus=num_gpus,
                engine=engine,
                concurrency=concurrency,
                metrics_endpoint=metrics_endpoint,
                metrics_target=metrics_target,
                topology_mode=topology_mode,
                session_routing=session_routing,
                session_header_name=session_header_name,
                cache_strategy=cache_strategy,
                cache_tier=cache_tier,
                cache_connector=cache_connector,
                session_affinity=session_affinity,
                request_rate=request_rate,
                arrival_model=arrival_model,
                arrival_shape=arrival_shape,
                warmup_requests=warmup_requests,
                goodput_slo=_parse_json_option(goodput_slo, option_name="goodput_slo"),
                strict_support=strict_support,
                synthetic_requests=synthetic_requests,
                synthetic_input_tokens=synthetic_input_tokens,
                synthetic_output_tokens=synthetic_output_tokens,
                synthetic_seed=synthetic_seed,
                context_file=context_file,
            )
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                "summary": f"Resolved benchmark plan for {workload_reference}",
                "run_plan": run_plan.model_dump(mode="json"),
                "support": support.model_dump(mode="json"),
            }
        )

    @app.command(name="benchmark")
    def benchmark_cmd(
        workload: Annotated[str, typer.Argument(help="Workload file path or built-in workload name")],
        endpoint: Annotated[str, typer.Argument(help="OpenAI-compatible request endpoint base URL")],
        experiment: Annotated[str, typer.Option(help="Optional built-in or file-backed experiment spec")] = "",
        model: Annotated[str, typer.Option(help="Override model name from the workload/experiment")] = "",
        gpu: Annotated[str, typer.Option(help="Concrete GPU SKU for support validation")] = "",
        num_gpus: Annotated[int | None, typer.Option(help="Concrete GPU count for support validation", min=1)] = None,
        engine: Annotated[str, typer.Option(help="Override engine for support validation")] = "",
        output: Annotated[Path | None, typer.Option(help="Where to write the benchmark artifact JSON")] = None,
        concurrency: Annotated[int | None, typer.Option(help="Override concurrency", min=1)] = None,
        metrics_endpoint: Annotated[str | None, typer.Option(help="Optional default Prometheus base URL")] = None,
        metrics_target: Annotated[
            list[str] | None,
            typer.Option(
                help="Additional metrics targets as name=url. Example: --metrics-target router=http://host:9000"
            ),
        ] = None,
        topology_mode: Annotated[str, typer.Option(help="Override topology mode")] = "",
        session_routing: Annotated[str, typer.Option(help="Override session routing mode")] = "",
        session_header_name: Annotated[str, typer.Option(help="Override session header name")] = "",
        cache_strategy: Annotated[str, typer.Option(help="Override cache strategy")] = "",
        cache_tier: Annotated[list[str] | None, typer.Option(help="Override cache tiers")] = None,
        cache_connector: Annotated[str, typer.Option(help="Override cache connector name")] = "",
        session_affinity: Annotated[bool | None, typer.Option(help="Override session affinity")] = None,
        request_rate: Annotated[float | None, typer.Option(help="Request rate for scheduled replay", min=0.0)] = None,
        arrival_model: Annotated[str, typer.Option(help="Arrival model: immediate, poisson, gamma")] = "immediate",
        arrival_shape: Annotated[float | None, typer.Option(help="Gamma arrival shape", min=0.0001)] = None,
        warmup_requests: Annotated[int, typer.Option(help="Warmup requests before measurement", min=0)] = 0,
        goodput_slo: Annotated[
            str,
            typer.Option(help="Optional JSON object for goodput thresholds"),
        ] = "",
        strict_support: Annotated[
            bool,
            typer.Option(
                "--strict-support/--no-strict-support",
                help="Fail benchmark execution on unsupported GPU/model/engine combinations",
            ),
        ] = True,
        synthetic_requests: Annotated[
            int | None,
            typer.Option(
                help="Procedurally expand the workload to this many requests",
                min=1,
            ),
        ] = None,
        synthetic_input_tokens: Annotated[
            int | None,
            typer.Option(
                help="Approximate input token target for procedural workloads",
                min=64,
            ),
        ] = None,
        synthetic_output_tokens: Annotated[
            int | None,
            typer.Option(
                help="Approximate output token target for procedural workloads",
                min=32,
            ),
        ] = None,
        synthetic_seed: Annotated[
            int,
            typer.Option(help="Seed for procedural workload expansion", min=0),
        ] = 42,
        context_file: Annotated[
            str,
            typer.Option(help=("Optional repo/context file used when procedurally expanding coding-long-context")),
        ] = "",
        provider: Annotated[
            str,
            typer.Option(help="Managed provider preset for auth defaults (fireworks, baseten, huggingface)"),
        ] = "",
        metrics_provider: Annotated[
            str,
            typer.Option(
                help=("Managed provider preset for the metrics endpoint if different from the request endpoint")
            ),
        ] = "",
        api_key: Annotated[
            str,
            typer.Option(envvar="OPENAI_API_KEY", help="API key or token for the request endpoint"),
        ] = "",
        auth_scheme: Annotated[str, typer.Option(help="Request auth scheme: bearer, api-key, x-api-key, raw")] = "",
        auth_header_name: Annotated[str, typer.Option(help="Override request auth header name")] = "",
        request_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional request headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        metrics_api_key: Annotated[
            str,
            typer.Option(envvar="INFERSCOPE_METRICS_API_KEY", help="API key for authenticated metrics endpoints"),
        ] = "",
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help="Metrics auth scheme: bearer, api-key, x-api-key, raw"),
        ] = "",
        metrics_auth_header_name: Annotated[str, typer.Option(help="Override metrics auth header name")] = "",
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help="Additional metrics headers as Header=Value. Repeat for multiple headers."),
        ] = None,
        capture_metrics: Annotated[
            bool,
            typer.Option(help="Capture Prometheus snapshots before and after the run"),
        ] = True,
    ):
        """Replay a workload pack against an OpenAI-compatible endpoint and save an artifact."""
        try:
            workload_reference, workload_pack, run_plan, support = _resolve_benchmark_plan(
                workload,
                endpoint,
                experiment=experiment,
                model=model,
                gpu=gpu,
                num_gpus=num_gpus,
                engine=engine,
                concurrency=concurrency,
                metrics_endpoint=metrics_endpoint,
                metrics_target=metrics_target,
                topology_mode=topology_mode,
                session_routing=session_routing,
                session_header_name=session_header_name,
                cache_strategy=cache_strategy,
                cache_tier=cache_tier,
                cache_connector=cache_connector,
                session_affinity=session_affinity,
                request_rate=request_rate,
                arrival_model=arrival_model,
                arrival_shape=arrival_shape,
                warmup_requests=warmup_requests,
                goodput_slo=_parse_json_option(goodput_slo, option_name="goodput_slo"),
                strict_support=strict_support,
                synthetic_requests=synthetic_requests,
                synthetic_input_tokens=synthetic_input_tokens,
                synthetic_output_tokens=synthetic_output_tokens,
                synthetic_seed=synthetic_seed,
                context_file=context_file,
            )
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        try:
            metrics_headers = parse_header_values(metrics_header, option_name="metrics header")
            request_headers = parse_header_values(request_header, option_name="request header")
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc

        artifact = asyncio.run(
            run_openai_replay(
                workload_pack,
                endpoint,
                metrics_endpoint=metrics_endpoint,
                run_plan=run_plan,
                workload_ref=workload_reference,
                api_key=(api_key or None),
                provider=provider,
                metrics_provider=metrics_provider,
                auth_scheme=auth_scheme,
                auth_header_name=auth_header_name,
                extra_headers=request_headers,
                metrics_api_key=(metrics_api_key or None),
                metrics_auth_scheme=metrics_auth_scheme,
                metrics_auth_header_name=metrics_auth_header_name,
                metrics_headers=metrics_headers,
                capture_metrics=capture_metrics,
                allow_private=True,
            )
        )
        artifact_path = output or build_default_artifact_path(artifact)
        saved_path = artifact.save_json(artifact_path)
        print_result(
            {
                "summary": (
                    f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                    f"requests succeeded | p95 latency={artifact.summary.latency_p95_ms:.1f} ms"
                    if artifact.summary.latency_p95_ms is not None
                    else (
                        f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                        "requests succeeded"
                    )
                ),
                "artifact_path": str(saved_path),
                "run_plan": run_plan.model_dump(mode="json"),
                "support": support.model_dump(mode="json"),
                "observed_runtime": (artifact.run_plan or {}).get("observed_runtime", {}),
                "benchmark": artifact.model_dump(mode="json"),
            }
        )

    @app.command(name="benchmark-compare")
    def benchmark_compare_cmd(
        baseline: Annotated[Path, typer.Argument(help="Baseline benchmark artifact JSON path")],
        candidate: Annotated[Path, typer.Argument(help="Candidate benchmark artifact JSON path")],
    ):
        """Compare two benchmark artifacts."""
        try:
            baseline_artifact = load_benchmark_artifact(baseline)
            candidate_artifact = load_benchmark_artifact(candidate)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc
        print_result(compare_benchmark_artifacts(baseline_artifact, candidate_artifact))
