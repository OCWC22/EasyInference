"""Engine-specific launch planning for benchmark experiments."""

from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from inferscope.benchmarks.catalog import load_experiment, load_workload
from inferscope.benchmarks.experiments import BenchmarkExperimentSpec, BenchmarkRunPlan, build_run_plan
from inferscope.engines.base import EngineConfig
from inferscope.hardware.gpu_profiles import get_gpu_profile
from inferscope.models.registry import get_model_variant
from inferscope.optimization.memory_planner import MemoryPlan
from inferscope.optimization.recommender import recommend
from inferscope.optimization.serving_profile import WorkloadMode


class GeneratedFile(BaseModel):
    """Generated config file content required for a launch plan."""

    model_config = ConfigDict(extra="forbid")

    path: str
    description: str
    content: str


class LaunchComponent(BaseModel):
    """One runnable component in a benchmark stack plan."""

    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    kind: Literal["engine", "router", "proxy"]
    engine: str | None = None
    command: str
    env_vars: dict[str, str] = Field(default_factory=dict)
    endpoint: str | None = None
    metrics_endpoint: str | None = None
    gpu_ids: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class BenchmarkStackPlan(BaseModel):
    """Production-oriented stack plan for launching a benchmark experiment."""

    model_config = ConfigDict(extra="forbid")

    experiment: str
    engine: str
    workload: str
    model: str
    gpu: str
    num_gpus: int
    host: str
    run_plan: dict
    benchmark_command: str
    components: list[LaunchComponent]
    generated_files: list[GeneratedFile] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class MaterializedStackFile(BaseModel):
    """One file written as part of a materialized benchmark stack."""

    model_config = ConfigDict(extra="forbid")

    path: str
    kind: Literal["plan", "config", "script", "env", "manifest", "readme"]
    description: str
    component: str | None = None


class MaterializedBenchmarkStack(BaseModel):
    """Concrete on-disk files for running a benchmark stack."""

    model_config = ConfigDict(extra="forbid")

    experiment: str
    output_dir: str
    plan_path: str
    manifest_path: str
    readme_path: str
    benchmark_script: str
    start_script: str
    stop_script: str
    files: list[MaterializedStackFile] = Field(default_factory=list)


def _map_workload_mode(workload_class: str) -> WorkloadMode:
    normalized = workload_class.strip().lower()
    if normalized in {"coding", "coding_agent"}:
        return WorkloadMode.CODING
    if normalized in {"agent", "tool_agent", "tool_calling_agent"}:
        return WorkloadMode.AGENT
    return WorkloadMode.CHAT


def _gpu_slices(num_gpus: int, topology_mode: str) -> dict[str, list[int]]:
    if topology_mode == "single_endpoint":
        return {"primary": list(range(num_gpus))}
    if num_gpus < 2:
        raise ValueError("Disaggregated benchmark experiments require at least 2 GPUs")
    prefill_count = max(1, num_gpus // 3)
    decode_count = num_gpus - prefill_count
    if decode_count < 1:
        raise ValueError("Need at least one decode GPU for disaggregated serving")
    return {
        "prefill": list(range(prefill_count)),
        "decode": list(range(prefill_count, prefill_count + decode_count)),
    }


def _compile_engine(
    model_name: str,
    gpu_name: str,
    num_gpus: int,
    workload_mode: WorkloadMode,
    engine: str,
) -> tuple[EngineConfig, MemoryPlan]:
    model_variant = get_model_variant(model_name)
    if model_variant is None:
        raise ValueError(f"Unknown model: {model_name}")
    gpu_profile = get_gpu_profile(gpu_name)
    if gpu_profile is None:
        raise ValueError(f"Unknown GPU: {gpu_name}")
    _, engine_config, memory_plan = recommend(
        model=model_variant,
        gpu=gpu_profile,
        num_gpus=num_gpus,
        workload=workload_mode,
        engine=engine,
    )
    return engine_config, memory_plan


def _append_command_args(command: str, *args: str) -> str:
    extra = " ".join(arg for arg in args if arg)
    return f"{command} {extra}".strip()


def _cuda_env(gpu_ids: list[int]) -> dict[str, str]:
    if not gpu_ids:
        return {}
    return {"CUDA_VISIBLE_DEVICES": ",".join(str(gpu_id) for gpu_id in gpu_ids)}


def _lmcache_config_files(host: str) -> list[GeneratedFile]:
    prefiller = GeneratedFile(
        path="lmcache-prefiller-config.yaml",
        description="LMCache config for the vLLM prefiller node",
        content=(
            "local_cpu: false\n"
            "enable_pd: true\n"
            'transfer_channel: "nixl"\n'
            'pd_role: "sender"\n'
            f'pd_proxy_host: "{host}"\n'
            "pd_proxy_port: 7500\n"
            "pd_buffer_size: 1073741824\n"
            'pd_buffer_device: "cuda"\n'
        ),
    )
    decoder = GeneratedFile(
        path="lmcache-decoder-config.yaml",
        description="LMCache config for the vLLM decoder node",
        content=(
            "local_cpu: false\n"
            "enable_pd: true\n"
            'transfer_channel: "nixl"\n'
            'pd_role: "receiver"\n'
            f'pd_peer_host: "{host}"\n'
            "pd_peer_init_port: 7300\n"
            "pd_peer_alloc_port: 7400\n"
            "pd_buffer_size: 1073741824\n"
            'pd_buffer_device: "cuda"\n'
        ),
    )
    return [prefiller, decoder]


def _dynamo_config_files(host: str) -> list[GeneratedFile]:
    content = f"""version: '1.0'
deployment:
  slo_planner:
    enabled: true
    target_latency_ms: 50
  kv_aware_router:
    enabled: true
  nixl:
    enabled: true
    bind_host: {host}
  grove:
    enabled: true
"""
    return [
        GeneratedFile(
            path="/tmp/dynamo-config.yaml",  # noqa: S108 - generated launcher path, not a runtime temp file write
            content=content,
            description="Dynamo 1.0 declarative configuration file with SLO Planner and KV-aware Router enabled.",
        )
    ]


def _default_ports(experiment: BenchmarkExperimentSpec) -> dict[str, int]:
    if experiment.engine == "vllm" and experiment.topology.mode == "prefill_decode_split":
        return {
            "primary": 9000,
            "prefill": 7100,
            "decode": 7200,
            "decode_init": 7300,
            "decode_alloc": 7400,
            "proxy_control": 7500,
        }
    if experiment.engine == "sglang" and experiment.topology.mode == "router_prefill_decode":
        return {
            "primary": 8000,
            "router_metrics": 29000,
            "prefill": 30000,
            "decode": 30001,
        }
    return {"primary": 8000}


def _build_benchmark_command(
    workload_reference: str,
    endpoint: str,
    experiment_name: str,
    run_plan: BenchmarkRunPlan,
) -> str:
    cmd_parts = [
        "inferscope",
        "benchmark",
        workload_reference,
        endpoint,
        "--experiment",
        experiment_name,
        "--model",
        run_plan.model,
    ]
    primary_target = next((target for target in run_plan.metrics_targets if target.role == "primary"), None)
    if primary_target is not None:
        cmd_parts.extend(["--metrics-endpoint", primary_target.endpoint])
    for target in run_plan.metrics_targets:
        if target.role == "primary":
            continue
        cmd_parts.extend(["--metrics-target", f"{target.name}={target.endpoint}"])
    return " ".join(cmd_parts)


def _resolve_env_value(value: str, generated_paths: set[str]) -> str:
    return f"${{STACK_ROOT}}/{value}" if value in generated_paths else value


def _render_component_script(component: LaunchComponent, env_file: str) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'cd "$STACK_ROOT"',
        "set -a",
        f'source "$STACK_ROOT/{env_file}"',
        "set +a",
        f"exec bash -lc {shlex.quote(component.command)}",
    ]
    return "\n".join(lines) + "\n"


def _render_start_script(component_scripts: list[tuple[str, str]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        'mkdir -p "$STACK_ROOT/logs" "$STACK_ROOT/pids"',
        "",
        "start_component() {",
        '  local name="$1"',
        '  local script="$2"',
        '  local pidfile="$STACK_ROOT/pids/${name}.pid"',
        '  if [[ -f "$pidfile" ]] && kill -0 "$(cat "$pidfile")" 2>/dev/null; then',
        '    echo "already running: ${name}"',
        "    return 0",
        "  fi",
        '  nohup "$script" >"$STACK_ROOT/logs/${name}.log" 2>&1 &',
        '  local pid="$!"',
        '  echo "$pid" >"$pidfile"',
        '  echo "started ${name} pid=${pid}"',
        "}",
        "",
    ]
    for name, script in component_scripts:
        lines.append(f'start_component {shlex.quote(name)} "$STACK_ROOT/{script}"')
    return "\n".join(lines) + "\n"


def _render_stop_script(component_names: list[str]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        'STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
        "",
        "stop_component() {",
        '  local name="$1"',
        '  local pidfile="$STACK_ROOT/pids/${name}.pid"',
        '  if [[ ! -f "$pidfile" ]]; then',
        '    echo "no pidfile for ${name}"',
        "    return 0",
        "  fi",
        '  local pid="$(cat "$pidfile")"',
        '  if kill -0 "$pid" 2>/dev/null; then',
        '    kill "$pid" || true',
        '    echo "stopped ${name} pid=${pid}"',
        "  fi",
        '  rm -f "$pidfile"',
        "}",
        "",
    ]
    for name in component_names:
        lines.append(f"stop_component {shlex.quote(name)}")
    return "\n".join(lines) + "\n"


def _render_benchmark_script(benchmark_command: str) -> str:
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'STACK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"',
            'cd "$STACK_ROOT"',
            f"exec bash -lc {shlex.quote(benchmark_command)}",
            "",
        ]
    )


def _render_stack_readme(stack_plan: BenchmarkStackPlan, component_scripts: list[tuple[LaunchComponent, str]]) -> str:
    lines = [
        f"# {stack_plan.experiment}",
        "",
        f"- Engine: `{stack_plan.engine}`",
        f"- Workload: `{stack_plan.workload}`",
        f"- Model: `{stack_plan.model}`",
        f"- GPU: `{stack_plan.gpu}` x `{stack_plan.num_gpus}`",
        f"- Host: `{stack_plan.host}`",
        "",
        "## Files",
        "",
        "- `stack-plan.json`: full generated plan",
        "- `manifest.json`: materialized file manifest",
        "- `scripts/start-all.sh`: launch all components in background",
        "- `scripts/stop-all.sh`: stop all started components",
        "- `scripts/run-benchmark.sh`: run the benchmark replay command",
        "",
        "## Components",
        "",
    ]
    for component, script_path in component_scripts:
        lines.extend(
            [
                f"### {component.name}",
                "",
                f"- Role: `{component.role}`",
                f"- Kind: `{component.kind}`",
                f"- Script: `{script_path}`",
                f"- Endpoint: `{component.endpoint or 'n/a'}`",
                f"- Metrics: `{component.metrics_endpoint or 'n/a'}`",
                "",
            ]
        )
        if component.notes:
            lines.append("Notes:")
            lines.extend(f"- {note}" for note in component.notes)
            lines.append("")
        if component.warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in component.warnings)
            lines.append("")
    if stack_plan.notes:
        lines.append("## Stack notes")
        lines.append("")
        lines.extend(f"- {note}" for note in stack_plan.notes)
        lines.append("")
    if stack_plan.warnings:
        lines.append("## Stack warnings")
        lines.append("")
        lines.extend(f"- {warning}" for warning in stack_plan.warnings)
        lines.append("")
    lines.extend(
        [
            "## Benchmark",
            "",
            f"```bash\n{stack_plan.benchmark_command}\n```",
            "",
        ]
    )
    return "\n".join(lines)


def materialize_benchmark_stack_plan(
    stack_plan: BenchmarkStackPlan,
    output_dir: str | Path,
    *,
    overwrite: bool = False,
) -> MaterializedBenchmarkStack:
    """Write a benchmark stack plan, scripts, and config files to disk."""
    root = Path(output_dir).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError(f"Output path must be a directory: {root}")
    if root.exists() and any(root.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {root}")
    if root.exists() and overwrite:
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    scripts_dir = root / "scripts"
    env_dir = root / "env"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    env_dir.mkdir(parents=True, exist_ok=True)

    written_files: list[MaterializedStackFile] = []
    generated_paths = {generated_file.path for generated_file in stack_plan.generated_files}
    placeholder_proxy_components = [
        component.name
        for component in stack_plan.components
        if component.kind == "proxy" and "<path-to-vllm-disagg-proxy>" in component.command
    ]
    if placeholder_proxy_components:
        names = ", ".join(placeholder_proxy_components)
        raise ValueError(
            "Cannot materialize a runnable bundle with placeholder proxy commands: "
            f"{names}. Rebuild the stack plan with a concrete vLLM proxy command."
        )

    plan_path = root / "stack-plan.json"
    plan_path.write_text(json.dumps(stack_plan.model_dump(mode="json"), indent=2) + "\n")
    written_files.append(MaterializedStackFile(path=str(plan_path), kind="plan", description="Serialized stack plan"))

    for generated_file in stack_plan.generated_files:
        file_path = root / generated_file.path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(generated_file.content)
        written_files.append(
            MaterializedStackFile(
                path=str(file_path),
                kind="config",
                description=generated_file.description,
            )
        )

    component_script_refs: list[tuple[LaunchComponent, str]] = []
    component_names: list[str] = []
    for component in stack_plan.components:
        component_names.append(component.name)
        env_path = env_dir / f"{component.name}.env"
        env_lines = [
            f"{key}={_resolve_env_value(value, generated_paths)}" for key, value in sorted(component.env_vars.items())
        ]
        env_path.write_text("\n".join(env_lines) + ("\n" if env_lines else ""))
        written_files.append(
            MaterializedStackFile(
                path=str(env_path),
                kind="env",
                description="Component environment variables",
                component=component.name,
            )
        )

        script_path = scripts_dir / f"{component.name}.sh"
        env_file = f"env/{component.name}.env"
        script_path.write_text(_render_component_script(component, env_file))
        script_path.chmod(0o755)
        written_files.append(
            MaterializedStackFile(
                path=str(script_path),
                kind="script",
                description="Launch script for component",
                component=component.name,
            )
        )
        component_script_refs.append((component, f"scripts/{component.name}.sh"))

    start_script = scripts_dir / "start-all.sh"
    start_script.write_text(
        _render_start_script([(component.name, script_path) for component, script_path in component_script_refs])
    )
    start_script.chmod(0o755)
    written_files.append(
        MaterializedStackFile(path=str(start_script), kind="script", description="Start all components")
    )

    stop_script = scripts_dir / "stop-all.sh"
    stop_script.write_text(_render_stop_script(list(reversed(component_names))))
    stop_script.chmod(0o755)
    written_files.append(MaterializedStackFile(path=str(stop_script), kind="script", description="Stop all components"))

    benchmark_script = scripts_dir / "run-benchmark.sh"
    benchmark_script.write_text(_render_benchmark_script(stack_plan.benchmark_command))
    benchmark_script.chmod(0o755)
    written_files.append(
        MaterializedStackFile(path=str(benchmark_script), kind="script", description="Run benchmark replay")
    )

    readme_path = root / "README.md"
    readme_path.write_text(_render_stack_readme(stack_plan, component_script_refs))
    written_files.append(
        MaterializedStackFile(path=str(readme_path), kind="readme", description="Stack bundle instructions")
    )

    manifest_path = root / "manifest.json"
    written_files.append(
        MaterializedStackFile(path=str(manifest_path), kind="manifest", description="Materialized file manifest")
    )
    manifest = {
        "experiment": stack_plan.experiment,
        "engine": stack_plan.engine,
        "files": [file.model_dump(mode="json") for file in written_files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    return MaterializedBenchmarkStack(
        experiment=stack_plan.experiment,
        output_dir=str(root),
        plan_path=str(plan_path),
        manifest_path=str(manifest_path),
        readme_path=str(readme_path),
        benchmark_script=str(benchmark_script),
        start_script=str(start_script),
        stop_script=str(stop_script),
        files=written_files,
    )


def build_benchmark_stack_plan(
    experiment_reference: str,
    gpu: str,
    num_gpus: int,
    *,
    model: str = "",
    host: str = "127.0.0.1",
    enable_trace: bool = False,
    otlp_endpoint: str = "",
    request_id_headers: list[str] | None = None,
    vllm_proxy_command: str = "",
) -> BenchmarkStackPlan:
    """Generate a concrete stack plan for a built-in benchmark experiment."""
    experiment = load_experiment(experiment_reference)
    workload_reference = experiment.workload
    request_id_headers = request_id_headers or ["x-request-id", "x-trace-id"]
    workload_pack = load_workload(workload_reference)
    selected_model = model or experiment.model or workload_pack.model or ""
    if not selected_model:
        raise ValueError("A model must be provided by the workload, experiment, or override")

    workload_mode = _map_workload_mode(workload_pack.workload_class)
    ports = _default_ports(experiment)
    gpu_layout = _gpu_slices(num_gpus, experiment.topology.mode)

    metrics_target_overrides = {}
    if experiment.engine == "vllm" and experiment.topology.mode == "prefill_decode_split":
        metrics_endpoint = f"http://{host}:{ports['decode']}"
        metrics_target_overrides = {
            "prefill": f"http://{host}:{ports['prefill']}",
            "decode": f"http://{host}:{ports['decode']}",
        }
    elif experiment.engine == "sglang" and experiment.topology.mode == "router_prefill_decode":
        metrics_endpoint = f"http://{host}:{ports['router_metrics']}"
        metrics_target_overrides = {
            "router": f"http://{host}:{ports['router_metrics']}",
            "prefill": f"http://{host}:{ports['prefill']}",
            "decode": f"http://{host}:{ports['decode']}",
        }
    else:
        metrics_endpoint = f"http://{host}:{ports['primary']}"

    request_endpoint = f"http://{host}:{ports['primary']}"
    run_plan = build_run_plan(
        workload_pack,
        request_endpoint,
        workload_ref=workload_reference,
        experiment=experiment,
        model=selected_model,
        concurrency=workload_pack.concurrency,
        metrics_endpoint=metrics_endpoint,
        metrics_target_overrides=metrics_target_overrides,
    )

    components: list[LaunchComponent] = []
    generated_files: list[GeneratedFile] = []
    notes: list[str] = []
    warnings: list[str] = []

    if experiment.engine == "vllm" and experiment.topology.mode == "single_endpoint":
        engine_config, memory_plan = _compile_engine(
            selected_model,
            gpu,
            num_gpus,
            workload_mode,
            "vllm",
        )
        component = LaunchComponent(
            name="vllm-primary",
            role="primary",
            kind="engine",
            engine="vllm",
            command=_append_command_args(
                engine_config.command,
                "--host",
                host,
                "--port",
                str(ports["primary"]),
            ),
            env_vars={**engine_config.env_vars, **_cuda_env(gpu_layout["primary"])},
            endpoint=request_endpoint,
            metrics_endpoint=metrics_endpoint,
            gpu_ids=gpu_layout["primary"],
            notes=list(engine_config.notes),
            warnings=list(engine_config.warnings),
        )
        if not memory_plan.fits:
            component.warnings.append("Model does not fit on the planned GPU slice")
            warnings.append("vLLM primary slice does not fit the selected model/GPU plan")
        components.append(component)
    elif experiment.engine == "vllm" and experiment.topology.mode == "prefill_decode_split":
        connector = experiment.topology.kv_connector or getattr(experiment.cache, "connector", None) or "NixlConnector"
        is_lmcache = connector == "LMCacheConnectorV1" or experiment.cache.strategy == "lmcache"

        if is_lmcache:
            generated_files = _lmcache_config_files(host)

        prefill_config, prefill_memory = _compile_engine(
            selected_model,
            gpu,
            len(gpu_layout["prefill"]),
            workload_mode,
            "vllm",
        )
        decode_config, decode_memory = _compile_engine(
            selected_model,
            gpu,
            len(gpu_layout["decode"]),
            workload_mode,
            "vllm",
        )

        if is_lmcache:
            prefill_transfer = json.dumps(
                {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_producer",
                    "kv_connector_extra_config": {
                        "discard_partial_chunks": False,
                        "lmcache_rpc_port": "producer1",
                    },
                }
            )
            decode_transfer = json.dumps(
                {
                    "kv_connector": "LMCacheConnectorV1",
                    "kv_role": "kv_consumer",
                    "kv_connector_extra_config": {
                        "discard_partial_chunks": False,
                        "lmcache_rpc_port": "consumer1",
                    },
                }
            )
            prefill_env = {
                **prefill_config.env_vars,
                **_cuda_env(gpu_layout["prefill"]),
                "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
                "LMCACHE_CONFIG_FILE": generated_files[0].path,
            }
            decode_env = {
                **decode_config.env_vars,
                **_cuda_env(gpu_layout["decode"]),
                "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
                "LMCACHE_CONFIG_FILE": generated_files[1].path,
            }
        else:
            prefill_transfer = json.dumps(
                {
                    "kv_connector": connector,
                    "kv_role": "kv_producer",
                }
            )
            decode_transfer = json.dumps(
                {
                    "kv_connector": connector,
                    "kv_role": "kv_consumer",
                }
            )
            prefill_env = {
                **prefill_config.env_vars,
                **_cuda_env(gpu_layout["prefill"]),
                "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
            }
            decode_env = {
                **decode_config.env_vars,
                **_cuda_env(gpu_layout["decode"]),
                "UCX_TLS": "cuda_ipc,cuda_copy,tcp",
            }

        components.extend(
            [
                LaunchComponent(
                    name="vllm-prefill",
                    role="prefill",
                    kind="engine",
                    engine="vllm",
                    command=_append_command_args(
                        prefill_config.command,
                        "--host",
                        host,
                        "--port",
                        str(ports["prefill"]),
                        "--disable-log-requests",
                        "--kv-transfer-config",
                        f"'{prefill_transfer}'",
                    ),
                    env_vars=prefill_env,
                    endpoint=f"http://{host}:{ports['prefill']}",
                    metrics_endpoint=f"http://{host}:{ports['prefill']}",
                    gpu_ids=gpu_layout["prefill"],
                    notes=list(prefill_config.notes),
                    warnings=list(prefill_config.warnings),
                ),
                LaunchComponent(
                    name="vllm-decode",
                    role="decode",
                    kind="engine",
                    engine="vllm",
                    command=_append_command_args(
                        decode_config.command,
                        "--host",
                        host,
                        "--port",
                        str(ports["decode"]),
                        "--disable-log-requests",
                        "--kv-transfer-config",
                        f"'{decode_transfer}'",
                    ),
                    env_vars=decode_env,
                    endpoint=f"http://{host}:{ports['decode']}",
                    metrics_endpoint=f"http://{host}:{ports['decode']}",
                    gpu_ids=gpu_layout["decode"],
                    notes=list(decode_config.notes),
                    warnings=list(decode_config.warnings),
                ),
                LaunchComponent(
                    name="vllm-disagg-proxy",
                    role="primary",
                    kind="proxy",
                    command=(
                        f"{vllm_proxy_command} "
                        f"--host {host} --port {ports['primary']} "
                        f"--prefiller-host {host} --prefiller-port {ports['prefill']} --num-prefillers 1 "
                        f"--decoder-host {host} --decoder-port {ports['decode']} "
                        f"--decoder-init-port {ports['decode_init']} --decoder-alloc-port {ports['decode_alloc']} "
                        f"--proxy-host {host} --proxy-port {ports['proxy_control']} --num-decoders 1"
                    )
                    if vllm_proxy_command
                    else (
                        "python3 <path-to-vllm-disagg-proxy> "
                        f"--host {host} --port {ports['primary']} "
                        f"--prefiller-host {host} --prefiller-port {ports['prefill']} --num-prefillers 1 "
                        f"--decoder-host {host} --decoder-port {ports['decode']} "
                        f"--decoder-init-port {ports['decode_init']} --decoder-alloc-port {ports['decode_alloc']} "
                        f"--proxy-host {host} --proxy-port {ports['proxy_control']} --num-decoders 1"
                    ),
                    endpoint=request_endpoint,
                    notes=[
                        "Use the proxy script shipped with the vLLM disaggregated-prefill examples.",
                        "Send benchmark traffic to the proxy endpoint, not directly to prefill/decode workers.",
                    ],
                    warnings=[]
                    if vllm_proxy_command
                    else ["Bundle is not runnable until a concrete vLLM disaggregation proxy command is provided."],
                ),
            ]
        )
        if not prefill_memory.fits:
            components[0].warnings.append("Model does not fit on the prefiller GPU slice")
            warnings.append("vLLM prefiller slice does not fit the selected model/GPU plan")
        if not decode_memory.fits:
            components[1].warnings.append("Model does not fit on the decoder GPU slice")
            warnings.append("vLLM decoder slice does not fit the selected model/GPU plan")
        notes.extend(
            [
                "LMCache config files are generated as part of this plan.",
                "Use NVLink or RDMA for realistic disaggregated-prefill performance.",
            ]
        )
    elif experiment.engine == "sglang" and experiment.topology.mode == "single_endpoint":
        engine_config, memory_plan = _compile_engine(
            selected_model,
            gpu,
            num_gpus,
            workload_mode,
            "sglang",
        )
        component = LaunchComponent(
            name="sglang-primary",
            role="primary",
            kind="engine",
            engine="sglang",
            command=_append_command_args(
                engine_config.command,
                "--host",
                host,
                "--port",
                str(ports["primary"]),
            ),
            env_vars={**engine_config.env_vars, **_cuda_env(gpu_layout["primary"])},
            endpoint=request_endpoint,
            metrics_endpoint=metrics_endpoint,
            gpu_ids=gpu_layout["primary"],
            notes=list(engine_config.notes),
            warnings=list(engine_config.warnings),
        )
        if not memory_plan.fits:
            component.warnings.append("Model does not fit on the planned GPU slice")
            warnings.append("SGLang primary slice does not fit the selected model/GPU plan")
        components.append(component)
    elif experiment.engine == "sglang" and experiment.topology.mode == "router_prefill_decode":
        prefill_config, prefill_memory = _compile_engine(
            selected_model,
            gpu,
            len(gpu_layout["prefill"]),
            workload_mode,
            "sglang",
        )
        decode_config, decode_memory = _compile_engine(
            selected_model,
            gpu,
            len(gpu_layout["decode"]),
            workload_mode,
            "sglang",
        )
        components.extend(
            [
                LaunchComponent(
                    name="sglang-prefill",
                    role="prefill",
                    kind="engine",
                    engine="sglang",
                    command=_append_command_args(
                        prefill_config.command,
                        "--host",
                        host,
                        "--port",
                        str(ports["prefill"]),
                        "--disaggregation-mode",
                        "prefill",
                        "--disaggregation-transfer-backend",
                        "nixl",
                    ),
                    env_vars={**prefill_config.env_vars, **_cuda_env(gpu_layout["prefill"])},
                    endpoint=f"http://{host}:{ports['prefill']}",
                    metrics_endpoint=f"http://{host}:{ports['prefill']}",
                    gpu_ids=gpu_layout["prefill"],
                    notes=list(prefill_config.notes),
                    warnings=list(prefill_config.warnings),
                ),
                LaunchComponent(
                    name="sglang-decode",
                    role="decode",
                    kind="engine",
                    engine="sglang",
                    command=_append_command_args(
                        decode_config.command,
                        "--host",
                        host,
                        "--port",
                        str(ports["decode"]),
                        "--disaggregation-mode",
                        "decode",
                        "--disaggregation-transfer-backend",
                        "nixl",
                    ),
                    env_vars={**decode_config.env_vars, **_cuda_env(gpu_layout["decode"])},
                    endpoint=f"http://{host}:{ports['decode']}",
                    metrics_endpoint=f"http://{host}:{ports['decode']}",
                    gpu_ids=gpu_layout["decode"],
                    notes=list(decode_config.notes),
                    warnings=list(decode_config.warnings),
                ),
            ]
        )
        router_command = (
            "python -m sglang_router.launch_router "
            f"--pd-disaggregation --prefill http://{host}:{ports['prefill']} --decode http://{host}:{ports['decode']} "
            "--prefill-policy cache_aware --decode-policy power_of_two "
            f"--host {host} --port {ports['primary']} "
            f"--prometheus-host {host} --prometheus-port {ports['router_metrics']}"
        )
        if request_id_headers:
            router_command += " --request-id-headers " + " ".join(request_id_headers)
        if enable_trace and otlp_endpoint:
            router_command += f" --enable-trace --otlp-traces-endpoint {otlp_endpoint}"
        components.append(
            LaunchComponent(
                name="sglang-router",
                role="primary",
                kind="router",
                command=router_command,
                endpoint=request_endpoint,
                metrics_endpoint=f"http://{host}:{ports['router_metrics']}",
                notes=[
                    "Enable request ID propagation for benchmark-to-worker correlation.",
                    "Router metrics are exposed separately from worker metrics.",
                ],
            )
        )
        if not prefill_memory.fits:
            components[0].warnings.append("Model does not fit on the prefiller GPU slice")
            warnings.append("SGLang prefiller slice does not fit the selected model/GPU plan")
        if not decode_memory.fits:
            components[1].warnings.append("Model does not fit on the decoder GPU slice")
            warnings.append("SGLang decoder slice does not fit the selected model/GPU plan")
        notes.extend(
            [
                "Cache-aware routing is enabled on the router by default.",
                "For trace export, provide --enable-trace and --otlp-endpoint.",
            ]
        )
    elif experiment.engine == "atom" and experiment.topology.mode == "single_endpoint":
        engine_config, memory_plan = _compile_engine(
            selected_model,
            gpu,
            num_gpus,
            workload_mode,
            "atom",
        )
        component = LaunchComponent(
            name="atom-primary",
            role="primary",
            kind="engine",
            engine="atom",
            command=_append_command_args(
                engine_config.command,
                "--host",
                host,
                "--port",
                str(ports["primary"]),
            ),
            env_vars={**engine_config.env_vars, **_cuda_env(gpu_layout["primary"])},
            endpoint=request_endpoint,
            metrics_endpoint=metrics_endpoint,
            gpu_ids=gpu_layout["primary"],
            notes=list(engine_config.notes),
            warnings=list(engine_config.warnings),
        )
        if not memory_plan.fits:
            component.warnings.append("Model does not fit on the planned GPU slice")
            warnings.append("ATOM primary slice does not fit the selected model/GPU plan")
        components.append(component)
    elif experiment.engine == "trtllm" and experiment.topology.mode == "single_endpoint":
        engine_config, memory_plan = _compile_engine(
            selected_model,
            gpu,
            num_gpus,
            workload_mode,
            "trtllm",
        )
        component = LaunchComponent(
            name="trtllm-primary",
            role="primary",
            kind="engine",
            engine="trtllm",
            command=_append_command_args(
                engine_config.command,
                "--host",
                host,
                "--port",
                str(ports["primary"]),
            ),
            env_vars={**engine_config.env_vars, **_cuda_env(gpu_layout["primary"])},
            endpoint=request_endpoint,
            metrics_endpoint=metrics_endpoint,
            gpu_ids=gpu_layout["primary"],
            notes=list(engine_config.notes),
            warnings=list(engine_config.warnings),
        )
        if not memory_plan.fits:
            component.warnings.append("Model does not fit on the planned GPU slice")
            warnings.append("TRT-LLM primary slice does not fit the selected model/GPU plan")
        components.append(component)
    elif experiment.engine == "dynamo" and experiment.topology.mode == "single_endpoint":
        generated_files.extend(_dynamo_config_files(host))
        engine_config, memory_plan = _compile_engine(
            selected_model,
            gpu,
            num_gpus,
            workload_mode,
            "dynamo",
        )
        component = LaunchComponent(
            name="dynamo-primary",
            role="primary",
            kind="engine",
            engine="dynamo",
            command=engine_config.command,
            env_vars={**engine_config.env_vars, **_cuda_env(gpu_layout["primary"])},
            endpoint=request_endpoint,
            metrics_endpoint=metrics_endpoint,
            gpu_ids=gpu_layout["primary"],
            notes=list(engine_config.notes),
            warnings=list(engine_config.warnings),
        )
        if not memory_plan.fits:
            component.warnings.append("Model does not fit on the planned GPU slice")
            warnings.append("Dynamo primary slice does not fit the selected model/GPU plan")
        components.append(component)
    else:
        raise ValueError(
            f"Unsupported experiment topology for launch planning: {experiment.engine}/{experiment.topology.mode}"
        )

    benchmark_command = _build_benchmark_command(
        workload_reference,
        request_endpoint,
        experiment.name,
        run_plan,
    )
    return BenchmarkStackPlan(
        experiment=experiment.name,
        engine=experiment.engine,
        workload=workload_reference,
        model=selected_model,
        gpu=gpu,
        num_gpus=num_gpus,
        host=host,
        run_plan=run_plan.model_dump(mode="json"),
        benchmark_command=benchmark_command,
        components=components,
        generated_files=generated_files,
        notes=notes,
        warnings=warnings,
    )
