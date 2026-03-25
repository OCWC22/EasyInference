"""Benchmark command registration for the InferScope CLI."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer

from inferscope.benchmarks import (
    build_benchmark_stack_plan,
    build_default_artifact_path,
    build_run_plan,
    compare_benchmark_artifacts,
    list_builtin_experiments,
    list_builtin_workloads,
    load_benchmark_artifact,
    load_experiment,
    load_workload,
    materialize_benchmark_stack_plan,
    parse_metrics_target_overrides,
    run_openai_replay,
)
from inferscope.endpoint_auth import parse_header_values


def _resolve_benchmark_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = '',
    model: str = '',
    concurrency: int | None = None,
    metrics_endpoint: str | None = None,
    metrics_target: list[str] | None = None,
    topology_mode: str = '',
    session_routing: str = '',
    session_header_name: str = '',
    cache_strategy: str = '',
    cache_tier: list[str] | None = None,
    cache_connector: str = '',
    session_affinity: bool | None = None,
):
    input_workload_pack = load_workload(workload)
    experiment_spec = load_experiment(experiment) if experiment else None
    if experiment_spec and input_workload_pack.name != experiment_spec.workload:
        raise ValueError(
            f"Workload '{input_workload_pack.name}' does not match experiment "
            f"'{experiment_spec.name}' workload '{experiment_spec.workload}'"
        )

    workload_reference = experiment_spec.workload if experiment_spec else workload
    workload_pack = load_workload(workload_reference) if experiment_spec else input_workload_pack
    metrics_target_overrides = parse_metrics_target_overrides(metrics_target)
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
    )
    return workload_reference, workload_pack, run_plan


def register_benchmark_commands(
    app: typer.Typer,
    *,
    print_result: Callable[[dict[str, Any]], None],
) -> None:
    """Register benchmark and evaluation commands on the main CLI app."""

    @app.command(name='benchmark-workloads')
    def benchmark_workloads_cmd():
        """List built-in benchmark workload packs."""
        workloads = list_builtin_workloads()
        print_result(
            {
                'summary': f'{len(workloads)} built-in workload pack(s) available',
                'workloads': workloads,
            }
        )

    @app.command(name='benchmark-experiments')
    def benchmark_experiments_cmd():
        """List built-in disaggregation/cache benchmark experiment specs."""
        experiments = list_builtin_experiments()
        print_result(
            {
                'summary': f'{len(experiments)} built-in benchmark experiment(s) available',
                'experiments': experiments,
            }
        )

    @app.command(name='benchmark-stack-plan')
    def benchmark_stack_plan_cmd(
        experiment: Annotated[str, typer.Argument(help='Built-in experiment spec name')],
        gpu: Annotated[str, typer.Argument(help='GPU type for the planned stack')],
        num_gpus: Annotated[int, typer.Option(help='Total GPU count available to the stack', min=1)] = 2,
        model: Annotated[str, typer.Option(help='Override model name from the experiment/workload')] = '',
        host: Annotated[str, typer.Option(help='Host/IP used in generated endpoints and commands')] = '127.0.0.1',
        enable_trace: Annotated[bool, typer.Option(help='Enable router OTLP tracing when supported')] = False,
        otlp_endpoint: Annotated[str, typer.Option(help='OTLP traces endpoint for router tracing')] = '',
        vllm_proxy_command: Annotated[
            str,
            typer.Option(help='Concrete proxy command for vLLM disaggregated-prefill bundles'),
        ] = '',
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
                request_id_headers=['x-request-id', 'x-trace-id'],
                vllm_proxy_command=vllm_proxy_command,
            )
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                'summary': f'Generated launch plan for {stack_plan.experiment}',
                'stack_plan': stack_plan.model_dump(mode='json'),
                'benchmark_command': stack_plan.benchmark_command,
            }
        )

    @app.command(name='benchmark-stack-write')
    def benchmark_stack_write_cmd(
        experiment: Annotated[str, typer.Argument(help='Built-in experiment spec name')],
        gpu: Annotated[str, typer.Argument(help='GPU type for the planned stack')],
        output_dir: Annotated[Path, typer.Argument(help='Directory to write scripts and configs into')],
        num_gpus: Annotated[int, typer.Option(help='Total GPU count available to the stack', min=1)] = 2,
        model: Annotated[str, typer.Option(help='Override model name from the experiment/workload')] = '',
        host: Annotated[str, typer.Option(help='Host/IP used in generated endpoints and commands')] = '127.0.0.1',
        enable_trace: Annotated[bool, typer.Option(help='Enable router OTLP tracing when supported')] = False,
        otlp_endpoint: Annotated[str, typer.Option(help='OTLP traces endpoint for router tracing')] = '',
        vllm_proxy_command: Annotated[
            str,
            typer.Option(help='Concrete proxy command for vLLM disaggregated-prefill bundles'),
        ] = '',
        overwrite: Annotated[bool, typer.Option(help='Allow writing into a non-empty output directory')] = False,
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
                request_id_headers=['x-request-id', 'x-trace-id'],
                vllm_proxy_command=vllm_proxy_command,
            )
            materialized = materialize_benchmark_stack_plan(stack_plan, output_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                'summary': f'Materialized runnable stack for {stack_plan.experiment}',
                'materialized': materialized.model_dump(mode='json'),
                'benchmark_command': stack_plan.benchmark_command,
            }
        )

    @app.command(name='benchmark-plan')
    def benchmark_plan_cmd(
        workload: Annotated[str, typer.Argument(help='Workload file path or built-in workload name')],
        endpoint: Annotated[str, typer.Argument(help='OpenAI-compatible request endpoint base URL')],
        experiment: Annotated[str, typer.Option(help='Optional built-in or file-backed experiment spec')] = '',
        model: Annotated[str, typer.Option(help='Override model name from the workload/experiment')] = '',
        concurrency: Annotated[int | None, typer.Option(help='Override concurrency', min=1)] = None,
        metrics_endpoint: Annotated[str | None, typer.Option(help='Optional default Prometheus base URL')] = None,
        metrics_target: Annotated[
            list[str] | None,
            typer.Option(
                help='Additional metrics targets as name=url. Example: --metrics-target router=http://host:9000'
            ),
        ] = None,
        topology_mode: Annotated[str, typer.Option(help='Override topology mode')] = '',
        session_routing: Annotated[str, typer.Option(help='Override session routing mode')] = '',
        session_header_name: Annotated[str, typer.Option(help='Override session header name')] = '',
        cache_strategy: Annotated[str, typer.Option(help='Override cache strategy')] = '',
        cache_tier: Annotated[list[str] | None, typer.Option(help='Override cache tiers')] = None,
        cache_connector: Annotated[str, typer.Option(help='Override cache connector name')] = '',
        session_affinity: Annotated[bool | None, typer.Option(help='Override session affinity')] = None,
    ):
        """Resolve workload + experiment + runtime overrides into a concrete benchmark run plan."""
        try:
            workload_reference, _, run_plan = _resolve_benchmark_plan(
                workload,
                endpoint,
                experiment=experiment,
                model=model,
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
            )
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        print_result(
            {
                'summary': f'Resolved benchmark plan for {workload_reference}',
                'run_plan': run_plan.model_dump(mode='json'),
            }
        )

    @app.command(name='benchmark')
    def benchmark_cmd(
        workload: Annotated[str, typer.Argument(help='Workload file path or built-in workload name')],
        endpoint: Annotated[str, typer.Argument(help='OpenAI-compatible request endpoint base URL')],
        experiment: Annotated[str, typer.Option(help='Optional built-in or file-backed experiment spec')] = '',
        model: Annotated[str, typer.Option(help='Override model name from the workload/experiment')] = '',
        output: Annotated[Path | None, typer.Option(help='Where to write the benchmark artifact JSON')] = None,
        concurrency: Annotated[int | None, typer.Option(help='Override concurrency', min=1)] = None,
        metrics_endpoint: Annotated[str | None, typer.Option(help='Optional default Prometheus base URL')] = None,
        metrics_target: Annotated[
            list[str] | None,
            typer.Option(
                help='Additional metrics targets as name=url. Example: --metrics-target router=http://host:9000'
            ),
        ] = None,
        topology_mode: Annotated[str, typer.Option(help='Override topology mode')] = '',
        session_routing: Annotated[str, typer.Option(help='Override session routing mode')] = '',
        session_header_name: Annotated[str, typer.Option(help='Override session header name')] = '',
        cache_strategy: Annotated[str, typer.Option(help='Override cache strategy')] = '',
        cache_tier: Annotated[list[str] | None, typer.Option(help='Override cache tiers')] = None,
        cache_connector: Annotated[str, typer.Option(help='Override cache connector name')] = '',
        session_affinity: Annotated[bool | None, typer.Option(help='Override session affinity')] = None,
        provider: Annotated[
            str,
            typer.Option(help='Managed provider preset for auth defaults (fireworks, baseten, huggingface)'),
        ] = '',
        metrics_provider: Annotated[
            str,
            typer.Option(
                help=(
                    'Managed provider preset for the metrics endpoint if different '
                    'from the request endpoint'
                )
            ),
        ] = '',
        api_key: Annotated[
            str,
            typer.Option(envvar='OPENAI_API_KEY', help='API key or token for the request endpoint'),
        ] = '',
        auth_scheme: Annotated[str, typer.Option(help='Request auth scheme: bearer, api-key, x-api-key, raw')] = '',
        auth_header_name: Annotated[str, typer.Option(help='Override request auth header name')] = '',
        request_header: Annotated[
            list[str] | None,
            typer.Option(help='Additional request headers as Header=Value. Repeat for multiple headers.'),
        ] = None,
        metrics_api_key: Annotated[
            str,
            typer.Option(envvar='INFERSCOPE_METRICS_API_KEY', help='API key for authenticated metrics endpoints'),
        ] = '',
        metrics_auth_scheme: Annotated[
            str,
            typer.Option(help='Metrics auth scheme: bearer, api-key, x-api-key, raw'),
        ] = '',
        metrics_auth_header_name: Annotated[str, typer.Option(help='Override metrics auth header name')] = '',
        metrics_header: Annotated[
            list[str] | None,
            typer.Option(help='Additional metrics headers as Header=Value. Repeat for multiple headers.'),
        ] = None,
        capture_metrics: Annotated[
            bool,
            typer.Option(help='Capture Prometheus snapshots before and after the run'),
        ] = True,
    ):
        """Replay a workload pack against an OpenAI-compatible endpoint and save an artifact."""
        try:
            workload_reference, workload_pack, run_plan = _resolve_benchmark_plan(
                workload,
                endpoint,
                experiment=experiment,
                model=model,
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
            )
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc

        try:
            metrics_headers = parse_header_values(metrics_header, option_name='metrics header')
            request_headers = parse_header_values(request_header, option_name='request header')
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
                'summary': (
                    f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                    f"requests succeeded | p95 latency={artifact.summary.latency_p95_ms:.1f} ms"
                    if artifact.summary.latency_p95_ms is not None
                    else (
                        f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                        'requests succeeded'
                    )
                ),
                'artifact_path': str(saved_path),
                'run_plan': run_plan.model_dump(mode='json'),
                'benchmark': artifact.model_dump(mode='json'),
            }
        )

    @app.command(name='benchmark-compare')
    def benchmark_compare_cmd(
        baseline: Annotated[Path, typer.Argument(help='Baseline benchmark artifact JSON path')],
        candidate: Annotated[Path, typer.Argument(help='Candidate benchmark artifact JSON path')],
    ):
        """Compare two benchmark artifacts."""
        try:
            baseline_artifact = load_benchmark_artifact(baseline)
            candidate_artifact = load_benchmark_artifact(candidate)
        except Exception as exc:  # noqa: BLE001
            raise typer.BadParameter(str(exc)) from exc
        print_result(compare_benchmark_artifacts(baseline_artifact, candidate_artifact))
