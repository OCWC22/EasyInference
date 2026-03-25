#!/usr/bin/env python3
"""generate_traces.py — Pre-generate all workload traces as JSONL files.

Reads each workload config from configs/workloads/*.yaml, instantiates the
appropriate WorkloadGenerator, generates traces with the configured seed,
and writes them as JSONL to a traces/ directory.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import click
import yaml

# ── Default workload generator mapping ───────────────────────────────────
# Maps workload_name -> (module_path, class_name)
_DEFAULT_GENERATORS: dict[str, tuple[str, str]] = {
    "chat": ("workloads.chat", "ChatWorkloadGenerator"),
    "agent": ("workloads.agent", "AgentTraceGenerator"),
    "rag": ("workloads.rag", "RAGTraceGenerator"),
    "coding": ("workloads.coding", "CodingTraceGenerator"),
}

# Default number of requests per workload trace
_DEFAULT_NUM_REQUESTS = 1000


def load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file."""
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def resolve_generator_class(workload_cfg: dict[str, Any], workload_name: str) -> type:
    """Resolve the generator class from config or defaults."""
    trace_cfg = workload_cfg.get("trace", {})
    generator_path = trace_cfg.get("generator", "")

    if generator_path:
        # e.g. "workloads.coding.CodingTraceGenerator"
        parts = generator_path.rsplit(".", 1)
        if len(parts) == 2:
            module_path, class_name = parts
        else:
            raise ValueError(
                f"Invalid generator path '{generator_path}' — "
                "expected 'module.ClassName'"
            )
    elif workload_name in _DEFAULT_GENERATORS:
        module_path, class_name = _DEFAULT_GENERATORS[workload_name]
    else:
        raise ValueError(
            f"No generator found for workload '{workload_name}'. "
            "Specify trace.generator in the workload config."
        )

    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def build_generator_kwargs(workload_cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract keyword arguments for the generator constructor from the config."""
    trace_cfg = workload_cfg.get("trace", {})
    kwargs: dict[str, Any] = {}

    # Seed
    seed = trace_cfg.get("seed", 42)
    kwargs["seed"] = int(seed)

    # ShareGPT path for chat workload
    if trace_cfg.get("type") == "sharegpt":
        source = trace_cfg.get("source", "")
        if source:
            kwargs["sharegpt_path"] = source

    # Turn filters for chat
    trace_filter = trace_cfg.get("filter", {})
    if "min_turns" in trace_filter:
        kwargs["min_turns"] = trace_filter["min_turns"]
    if "max_turns" in trace_filter:
        kwargs["max_turns"] = trace_filter["max_turns"]

    return kwargs


def generate_trace_for_workload(
    workload_cfg: dict[str, Any],
    workload_name: str,
    num_requests: int,
    output_dir: Path,
) -> Path:
    """Generate a single workload trace and save it."""
    cls = resolve_generator_class(workload_cfg, workload_name)
    kwargs = build_generator_kwargs(workload_cfg)

    click.echo(f"  Generator : {cls.__module__}.{cls.__name__}")
    click.echo(f"  Seed      : {kwargs.get('seed', 42)}")
    click.echo(f"  Requests  : {num_requests}")

    generator = cls(**kwargs)
    requests = generator.generate(num_requests)

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{workload_name}.jsonl"
    generator.save(requests, trace_path)

    click.echo(f"  Saved     : {trace_path} ({len(requests)} requests)")
    return trace_path


@click.command("generate-traces")
@click.option(
    "--config-root",
    type=click.Path(exists=True),
    default="configs",
    help="Root directory for ISB-1 config files.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="traces",
    help="Output directory for generated traces.",
)
@click.option(
    "--num-requests",
    type=int,
    default=_DEFAULT_NUM_REQUESTS,
    help=f"Number of requests per workload (default: {_DEFAULT_NUM_REQUESTS}).",
)
@click.option(
    "--workload",
    "workload_filter",
    type=str,
    default=None,
    help="Generate traces for a specific workload only.",
)
@click.option("--dry-run", is_flag=True, help="Show what would be generated without writing.")
def main(
    config_root: str,
    output_dir: str,
    num_requests: int,
    workload_filter: str | None,
    dry_run: bool,
) -> None:
    """Pre-generate all ISB-1 workload traces as JSONL files."""
    root = Path(config_root).resolve()
    workload_dir = root / "workloads"
    out = Path(output_dir).resolve()

    if not workload_dir.is_dir():
        click.echo(f"Workload config directory not found: {workload_dir}", err=True)
        raise SystemExit(1)

    # Discover workload configs
    workload_paths = sorted(workload_dir.glob("*.yaml"))
    if not workload_paths:
        click.echo("No workload configs found.", err=True)
        raise SystemExit(1)

    click.echo(f"Found {len(workload_paths)} workload config(s)")
    click.echo(f"Output directory: {out}")
    click.echo(f"Requests per workload: {num_requests}")
    click.echo("")

    generated = 0
    skipped = 0
    errors = 0

    for wp in workload_paths:
        workload_cfg = load_yaml(wp)
        workload_name = workload_cfg.get("workload_name", wp.stem)

        # Apply filter
        if workload_filter and workload_name != workload_filter:
            continue

        click.echo(f"── {workload_name} ({wp.name}) ──────────────────")

        if dry_run:
            trace_cfg = workload_cfg.get("trace", {})
            click.echo(f"  [DRY RUN] Would generate {num_requests} requests")
            click.echo(f"  Seed: {trace_cfg.get('seed', 42)}")
            click.echo(f"  Output: {out / f'{workload_name}.jsonl'}")
            generated += 1
            continue

        try:
            generate_trace_for_workload(workload_cfg, workload_name, num_requests, out)
            generated += 1
        except Exception as exc:
            click.echo(f"  ERROR: {exc}", err=True)
            errors += 1

    click.echo("")
    click.echo("────────────────────────────────────────────────")
    click.echo(f"Trace generation complete.")
    click.echo(f"  Generated : {generated}")
    click.echo(f"  Errors    : {errors}")
    click.echo(f"  Output    : {out}")
    click.echo("────────────────────────────────────────────────")

    if errors > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
