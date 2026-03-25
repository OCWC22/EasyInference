"""Benchmark MCP tool registration for the InferScope server."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastmcp import FastMCP

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
    run_openai_replay,
)
from inferscope.config import settings
from inferscope.endpoint_auth import resolve_auth_payload


def _resolve_artifact_path_for_mcp(path_or_name: str) -> Path:
    """Resolve an artifact path under the benchmark directory only."""
    artifact_root = settings.benchmark_dir.resolve()
    candidate = Path(path_or_name)
    if not candidate.is_absolute():
        candidate = artifact_root / candidate
    resolved = candidate.resolve()
    if artifact_root not in resolved.parents and resolved != artifact_root:
        raise ValueError(f'Artifact path must stay under {artifact_root}')
    return resolved


def _default_stack_bundle_dir(experiment: str, gpu: str, num_gpus: int, model: str = '') -> Path:
    model_part = f'-{model}' if model else ''
    raw_name = f'{experiment}-{gpu}-{num_gpus}gpus{model_part}'
    safe_name = ''.join(char if char.isalnum() or char in {'-', '_'} else '-' for char in raw_name)
    return settings.benchmark_dir / 'stacks' / safe_name


def _resolve_benchmark_plan(
    workload: str,
    endpoint: str,
    *,
    experiment: str = '',
    model: str = '',
    metrics_endpoint: str = '',
    concurrency: int = 0,
    metrics_target_overrides: dict[str, str] | None = None,
):
    try:
        input_workload_pack = load_workload(workload)
        experiment_spec = load_experiment(experiment) if experiment else None
        if experiment_spec and input_workload_pack.name != experiment_spec.workload:
            return {
                'error': (
                    f"Workload '{input_workload_pack.name}' does not match experiment "
                    f"'{experiment_spec.name}' workload '{experiment_spec.workload}'"
                ),
                'summary': f'❌ Workload does not match experiment: {input_workload_pack.name}',
                'confidence': 1.0,
                'evidence': 'benchmark_plan_resolution',
            }, None, None, None

        workload_reference = experiment_spec.workload if experiment_spec else workload
        workload_pack = load_workload(workload_reference) if experiment_spec else input_workload_pack
        run_plan = build_run_plan(
            workload_pack,
            endpoint,
            workload_ref=workload_reference,
            experiment=experiment_spec,
            model=(model or None),
            concurrency=(concurrency or None),
            metrics_endpoint=(metrics_endpoint or None),
            metrics_target_overrides=metrics_target_overrides or {},
        )
    except Exception as exc:  # noqa: BLE001
        return {
            'error': str(exc),
            'summary': '❌ Failed to resolve benchmark plan',
            'confidence': 1.0,
            'evidence': 'benchmark_plan_resolution',
        }, None, None, None

    return None, workload_reference, workload_pack, run_plan


def register_benchmark_tools(mcp: FastMCP) -> None:
    """Register evaluation and benchmark MCP tools."""

    @mcp.tool()
    async def tool_list_benchmark_workloads() -> dict[str, Any]:
        """List packaged workload packs for coding, RAG, agents, and mixed tenancy."""
        workloads = list_builtin_workloads()
        return {
            'summary': f'{len(workloads)} built-in workload pack(s) available',
            'workloads': workloads,
            'confidence': 1.0,
            'evidence': 'packaged_workload_catalog',
        }

    @mcp.tool()
    async def tool_list_benchmark_experiments() -> dict[str, Any]:
        """List packaged experiments for colocated and disaggregated cache-aware deployments."""
        experiments = list_builtin_experiments()
        return {
            'summary': f'{len(experiments)} built-in benchmark experiment(s) available',
            'experiments': experiments,
            'confidence': 1.0,
            'evidence': 'packaged_experiment_catalog',
        }

    @mcp.tool()
    async def tool_generate_benchmark_stack_plan(
        experiment: str,
        gpu: str,
        num_gpus: int = 2,
        model: str = '',
        host: str = '127.0.0.1',
        enable_trace: bool = False,
        otlp_endpoint: str = '',
        vllm_proxy_command: str = '',
    ) -> dict[str, Any]:
        """Generate live launch commands for a packaged vLLM or SGLang benchmark stack."""
        available_experiments = list_builtin_experiments()
        if experiment not in available_experiments:
            return {
                'error': f"Unknown built-in experiment '{experiment}'",
                'available_experiments': available_experiments,
                'summary': f'❌ Unknown built-in experiment: {experiment}',
                'confidence': 1.0,
                'evidence': 'builtin_experiment_catalog',
            }
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
            return {
                'error': str(exc),
                'summary': '❌ Failed to generate benchmark stack plan',
                'confidence': 1.0,
                'evidence': 'benchmark_stack_plan',
            }
        return {
            'summary': f'Generated launch plan for {stack_plan.experiment}',
            'stack_plan': stack_plan.model_dump(mode='json'),
            'benchmark_command': stack_plan.benchmark_command,
            'confidence': 0.9,
            'evidence': 'benchmark_stack_plan',
        }

    @mcp.tool()
    async def tool_materialize_benchmark_stack_plan(
        experiment: str,
        gpu: str,
        num_gpus: int = 2,
        model: str = '',
        host: str = '127.0.0.1',
        enable_trace: bool = False,
        otlp_endpoint: str = '',
        vllm_proxy_command: str = '',
        output_dir: str = '',
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Write a runnable benchmark stack bundle under the benchmark artifact directory."""
        available_experiments = list_builtin_experiments()
        if experiment not in available_experiments:
            return {
                'error': f"Unknown built-in experiment '{experiment}'",
                'available_experiments': available_experiments,
                'summary': f'❌ Unknown built-in experiment: {experiment}',
                'confidence': 1.0,
                'evidence': 'builtin_experiment_catalog',
            }
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
            bundle_dir = (
                _resolve_artifact_path_for_mcp(output_dir)
                if output_dir
                else _default_stack_bundle_dir(experiment, gpu, num_gpus, model).resolve()
            )
            materialized = materialize_benchmark_stack_plan(stack_plan, bundle_dir, overwrite=overwrite)
        except Exception as exc:  # noqa: BLE001
            return {
                'error': str(exc),
                'summary': '❌ Failed to materialize benchmark stack plan',
                'confidence': 1.0,
                'evidence': 'benchmark_stack_materialization',
            }
        return {
            'summary': f'Materialized runnable stack for {stack_plan.experiment}',
            'materialized': materialized.model_dump(mode='json'),
            'benchmark_command': stack_plan.benchmark_command,
            'confidence': 0.95,
            'evidence': 'benchmark_stack_materialization',
        }

    @mcp.tool()
    async def tool_resolve_benchmark_plan(
        workload: str,
        endpoint: str,
        experiment: str = '',
        model: str = '',
        metrics_endpoint: str = '',
        concurrency: int = 0,
        metrics_target_overrides: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Resolve a workload reference and optional experiment reference into a concrete run plan."""
        error, workload_reference, _, run_plan = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            model=model,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
        )
        if error is not None:
            return error
        return {
            'summary': f'Resolved benchmark plan for {workload_reference}',
            'run_plan': run_plan.model_dump(mode='json'),
            'confidence': 0.95,
            'evidence': 'benchmark_plan_resolution',
        }

    @mcp.tool()
    async def tool_run_benchmark(
        workload: str,
        endpoint: str,
        experiment: str = '',
        model: str = '',
        metrics_endpoint: str = '',
        concurrency: int = 0,
        capture_metrics: bool = True,
        save_artifact: bool = True,
        metrics_target_overrides: dict[str, str] | None = None,
        provider: str = '',
        metrics_provider: str = '',
        request_auth: dict | None = None,
        metrics_auth: dict | None = None,
    ) -> dict[str, Any]:
        """Replay a workload reference against an OpenAI-compatible endpoint."""
        error, workload_reference, workload_pack, run_plan = _resolve_benchmark_plan(
            workload,
            endpoint,
            experiment=experiment,
            model=model,
            metrics_endpoint=metrics_endpoint,
            concurrency=concurrency,
            metrics_target_overrides=metrics_target_overrides,
        )
        if error is not None:
            return error

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
                auth_scheme=request_auth_config.auth_scheme if request_auth_config else '',
                auth_header_name=request_auth_config.auth_header_name if request_auth_config else '',
                extra_headers=request_auth_config.headers if request_auth_config else None,
                metrics_api_key=(metrics_auth_config.api_key or None) if metrics_auth_config else None,
                metrics_auth_scheme=metrics_auth_config.auth_scheme if metrics_auth_config else '',
                metrics_auth_header_name=(
                    metrics_auth_config.auth_header_name if metrics_auth_config else ''
                ),
                metrics_headers=metrics_auth_config.headers if metrics_auth_config else None,
                capture_metrics=capture_metrics,
                allow_private=False,
            )
            artifact_path = ''
            if save_artifact:
                artifact_path = str(artifact.save_json(build_default_artifact_path(artifact)))
        except Exception as exc:  # noqa: BLE001
            return {
                'error': str(exc),
                'summary': '❌ Benchmark run failed',
                'confidence': 1.0,
                'evidence': 'live_benchmark_replay',
            }
        return {
            'summary': (
                f"Benchmark completed: {artifact.summary.succeeded}/{artifact.summary.total_requests} "
                'requests succeeded'
            ),
            'artifact_path': artifact_path,
            'benchmark_id': artifact.benchmark_id,
            'run_plan': run_plan.model_dump(mode='json'),
            'benchmark_summary': artifact.summary.model_dump(mode='json'),
            'confidence': 0.85,
            'evidence': 'live_benchmark_replay',
        }

    @mcp.tool()
    async def tool_compare_benchmarks(baseline_artifact: str, candidate_artifact: str) -> dict[str, Any]:
        """Compare two saved benchmark artifacts and report latency/TTFT deltas."""
        baseline = load_benchmark_artifact(_resolve_artifact_path_for_mcp(baseline_artifact))
        candidate = load_benchmark_artifact(_resolve_artifact_path_for_mcp(candidate_artifact))
        comparison = compare_benchmark_artifacts(baseline, candidate)
        comparison['confidence'] = 0.9
        comparison['evidence'] = 'benchmark_artifact_comparison'
        return comparison

    @mcp.tool()
    async def tool_get_benchmark_artifact(artifact_name: str) -> dict[str, Any]:
        """Read a saved benchmark artifact by filename from the benchmark directory."""
        artifact = load_benchmark_artifact(_resolve_artifact_path_for_mcp(artifact_name))
        return {
            'summary': f'Loaded benchmark artifact {artifact.default_filename}',
            'artifact': artifact.model_dump(mode='json'),
            'confidence': 1.0,
            'evidence': 'saved_benchmark_artifact',
        }
