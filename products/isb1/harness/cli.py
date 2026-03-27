"""ISB-1 Benchmark Harness CLI — unified command-line interface."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click

from harness.paths import (
    default_config_root,
    default_results_root,
    resolve_existing_path,
    resolve_path,
)

logger = logging.getLogger(__name__)


def _resolve_existing_click_path(
    _ctx: click.Context, _param: click.Parameter, value: Path | None
) -> Path | None:
    if value is None:
        return None
    try:
        return resolve_existing_path(value)
    except FileNotFoundError as exc:
        raise click.BadParameter(str(exc)) from exc


def _resolve_click_path(
    _ctx: click.Context, _param: click.Parameter, value: Path | None
) -> Path | None:
    if value is None:
        return None
    return resolve_path(value)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging.")
@click.version_option(version="1.0.0", prog_name="isb1")
def main(verbose: bool) -> None:
    """ISB-1: Inference Serving Benchmark Standard 1."""
    _setup_logging(verbose)


# ── validate ─────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--sweep",
    "sweep_path",
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option("--all-yaml", is_flag=True, help="Parse-check every YAML under config root.")
def validate(sweep_path: str | None, config_root: str, all_yaml: bool) -> None:
    """Validate ISB-1 configuration files."""
    from harness.config_validator import ConfigValidator

    validator = ConfigValidator(config_root)

    if all_yaml:
        click.echo("Checking all YAML files...")
        res = validator.validate_all_yamls()
        click.echo(res.summary())

    if sweep_path:
        click.echo(f"Validating sweep: {sweep_path}")
        res = validator.validate_sweep(sweep_path)
        click.echo(res.summary())
        if not res.ok:
            raise SystemExit(1)
        click.echo("Sweep validation passed.")

    if not sweep_path and not all_yaml:
        click.echo("No action specified. Use --sweep or --all-yaml.")
        raise SystemExit(1)


# ── plan ─────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config",
    "--sweep",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
def plan(config_path: str, config_root: str) -> None:
    """Print the sweep matrix without executing."""
    from harness.sweep import SweepOrchestrator

    orchestrator = SweepOrchestrator(
        sweep_path=config_path,
        config_root=config_root,
        dry_run=True,
    )
    click.echo(orchestrator.plan())


# ── run ──────────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--config",
    "--sweep",
    "config_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to sweep config YAML.",
)
@click.option(
    "--output",
    "output_dir",
    default=default_results_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Output directory. Defaults to the product-local results/ tree.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option("--dry-run", is_flag=True, help="Print plan without executing.")
@click.option("--resume", "do_resume", is_flag=True, help="Resume a previous sweep.")
def run(
    config_path: str,
    output_dir: str,
    config_root: str,
    dry_run: bool,
    do_resume: bool,
) -> None:
    """Execute an ISB-1 benchmark sweep."""
    from harness.sweep import SweepOrchestrator

    orchestrator = SweepOrchestrator(
        sweep_path=config_path,
        output_dir=output_dir,
        config_root=config_root,
        dry_run=dry_run,
    )

    if dry_run:
        click.echo(orchestrator.plan())
        return

    if do_resume:
        summary = orchestrator.resume()
    else:
        summary = orchestrator.execute()

    click.echo("\nSweep complete:")
    click.echo(f"  Total cells:  {summary.total_cells}")
    click.echo(f"  Completed:    {summary.completed}")
    click.echo(f"  Failed:       {summary.failed}")
    click.echo(f"  Skipped:      {summary.skipped}")

    if summary.failed > 0:
        raise SystemExit(1)


# ── run-cell ─────────────────────────────────────────────────────────────


@main.command("run-cell")
@click.option("--gpu", required=True, help="GPU short name (e.g. h100).")
@click.option("--model", required=True, help="Model short name (e.g. llama70b).")
@click.option("--workload", required=True, help="Workload name (e.g. chat).")
@click.option("--mode", required=True, help="Mode name (e.g. mode_a).")
@click.option("--quantization", default="fp8", help="Quantization format.")
@click.option("--trial", default=1, type=int, help="Trial number.")
@click.option("--gpu-count", default=None, type=int, help="Number of GPUs (auto-detected if omitted).")
@click.option(
    "--output",
    "output_dir",
    default=default_results_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Output directory. Defaults to the product-local results/ tree.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--endpoint",
    default=None,
    help="External endpoint URL (e.g. https://my-modal-app.modal.run). "
    "Skips local vLLM server launch and GPU telemetry.",
)
def run_cell(
    gpu: str,
    model: str,
    workload: str,
    mode: str,
    quantization: str,
    trial: int,
    gpu_count: int | None,
    output_dir: str,
    config_root: str,
    endpoint: str | None,
) -> None:
    """Execute a single benchmark cell."""
    from harness.config_validator import ConfigValidator
    from harness.runner import BenchmarkRunner, CellConfig

    validator = ConfigValidator(config_root)

    # Resolve model info
    try:
        model_cfg = validator.load_model(model)
    except FileNotFoundError:
        click.echo(f"ERROR: Model config for '{model}' not found.", err=True)
        raise SystemExit(1)

    model_hf_id = model_cfg.get("hf_model_id", "")

    # Auto-detect gpu_count if not provided
    if gpu_count is None:
        quant_key = "fp8" if quantization.startswith("fp8") else quantization
        min_gpus = model_cfg.get("min_gpus", {})
        quant_map = min_gpus.get(quant_key, min_gpus.get("bf16", {}))
        gpu_count = quant_map.get(gpu, 1)

    # Resolve topology
    rec = model_cfg.get("recommended_topology", {})
    quant_key = "fp8" if quantization.startswith("fp8") else quantization
    topology = rec.get(quant_key, rec.get("bf16", {})).get(gpu, f"tp{gpu_count}")

    # Resolve workload
    try:
        wl_cfg = validator.load_workload(workload)
    except FileNotFoundError:
        click.echo(f"ERROR: Workload config for '{workload}' not found.", err=True)
        raise SystemExit(1)

    rate_sweep = wl_cfg.get("arrival", {}).get("rate_sweep", [1.0])
    num_prompts = int(wl_cfg.get("trace", {}).get("num_requests", 1000))
    arrival_cfg = wl_cfg.get("arrival", {})
    arrival_model = str(arrival_cfg.get("model", "poisson"))
    arrival_shape = float(arrival_cfg["shape"]) if "shape" in arrival_cfg else None
    goodput_slo = wl_cfg.get("slo") or None

    cell = CellConfig(
        gpu=gpu,
        gpu_count=gpu_count,
        model=model,
        model_hf_id=model_hf_id,
        workload=workload,
        mode=mode,
        quantization=quantization,
        topology=topology,
        trial_number=trial,
        num_prompts=num_prompts,
        rate_sweep=rate_sweep,
        seed=42 + trial,
        arrival_model=arrival_model,
        arrival_shape=arrival_shape,
        goodput_slo=goodput_slo,
        output_dir=output_dir,
        config_root=config_root,
        external_endpoint=endpoint,
    )

    runner = BenchmarkRunner(cell)
    result = runner.run()

    click.echo(f"\nRun complete: {result.run_id}")
    click.echo(f"  Status: {result.status}")
    if result.error_message:
        click.echo(f"  Error: {result.error_message}")
    click.echo(f"  Manifest: {result.manifest_path}")

    if result.status == "failed":
        raise SystemExit(1)


# ── analyze ──────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(),
    help="Output file for analysis results (JSON).",
)
def analyze(results_dir: str, output_path: str | None) -> None:
    """Analyze benchmark results and compute metrics."""
    from analysis.metrics import MetricComputer

    results_path = Path(results_dir)

    all_metrics: list[dict] = []
    for manifest_path in sorted(results_path.rglob("manifest.json")):
        run_dir = manifest_path.parent
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest_data.get("status") != "completed":
            continue

        # Load benchmark results
        bench_dir = run_dir / "benchmark"
        if not bench_dir.exists():
            continue

        json_files = sorted(bench_dir.glob("*.json"))
        if not json_files:
            continue

        raw = json.loads(json_files[-1].read_text(encoding="utf-8"))
        per_request = raw.get("per_request", [])

        # Load engine metrics
        em_path = run_dir / "engine_metrics.jsonl"
        engine_data: list[dict] = []
        if em_path.exists():
            for line in em_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    engine_data.append(json.loads(line))

        # Load telemetry
        telem_data: list[dict] = []
        telem_path = run_dir / "telemetry.csv"
        if telem_path.exists():
            import csv

            with open(telem_path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        telem_data.append(
                            {
                                "power_watts": float(row.get("power_draw_watts", 0) or 0),
                                "timestamp": float(row.get("timestamp", 0) or 0),
                            }
                        )
                    except (ValueError, TypeError):
                        pass

        computer = MetricComputer(
            gpu_name=manifest_data.get("gpu", ""),
            gpu_count=manifest_data.get("gpu_count", 1),
        )
        metrics = computer.compute(per_request, engine_data, telem_data)
        entry = {**manifest_data, **metrics.to_dict()}
        all_metrics.append(entry)

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(all_metrics, indent=2) + "\n", encoding="utf-8"
        )
        click.echo(f"Analysis written to {output_path} ({len(all_metrics)} cells)")
    else:
        click.echo(json.dumps(all_metrics, indent=2))


# ── claims ───────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
def claims(results_dir: str) -> None:
    """Extract publishable performance claims from results."""
    results_path = Path(results_dir)
    manifests: list[dict] = []

    for mp in sorted(results_path.rglob("manifest.json")):
        data = json.loads(mp.read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            manifests.append(data)

    if not manifests:
        click.echo("No completed runs found.")
        return

    click.echo(f"Found {len(manifests)} completed runs.")
    click.echo("\nPublishable claims require:")
    click.echo("  - Minimum 3 trials per configuration")
    click.echo("  - CV < 10% across trials")
    click.echo("  - Stable warmup achieved")
    click.echo("\nRun 'isb1 analyze' first to compute full metrics.")


# ── leaderboard ──────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--analysis",
    "--input",
    "analysis_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to analysis JSON output.",
)
@click.option("--sort-by", default="generation_throughput", help="Metric to sort by.")
@click.option("--top", default=20, type=int, help="Number of entries to show.")
def leaderboard(analysis_path: str, sort_by: str, top: int) -> None:
    """Display a leaderboard from analysis results."""
    data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    if not data:
        click.echo("No data to display.")
        return

    # Sort by the specified metric (descending)
    try:
        sorted_data = sorted(data, key=lambda x: x.get(sort_by, 0), reverse=True)
    except TypeError:
        click.echo(f"Cannot sort by '{sort_by}'.")
        return

    header = (
        f"{'Rank':<5} {'GPU':<8} {'Model':<15} {'Workload':<10} "
        f"{'Quant':<6} {sort_by:<25} {'Status':<10}"
    )
    click.echo(header)
    click.echo("-" * len(header))

    for i, entry in enumerate(sorted_data[:top], 1):
        val = entry.get(sort_by, "N/A")
        if isinstance(val, float):
            val_str = f"{val:.2f}"
        else:
            val_str = str(val)
        click.echo(
            f"{i:<5} {entry.get('gpu', ''):<8} {entry.get('model', ''):<15} "
            f"{entry.get('workload', ''):<10} {entry.get('quantization', ''):<6} "
            f"{val_str:<25} {entry.get('status', ''):<10}"
        )


# ── report ───────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--analysis",
    "--input",
    "analysis_path",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to analysis JSON output.",
)
@click.option(
    "--output",
    "output_path",
    default="report.html",
    type=click.Path(),
    help="Output HTML report path.",
)
@click.option(
    "--template",
    default=None,
    type=click.Path(path_type=Path),
    callback=_resolve_click_path,
    help="Jinja2 template file.",
)
def report(analysis_path: str, output_path: str, template: str | None) -> None:
    """Generate an HTML report from analysis results."""
    import jinja2

    data = json.loads(Path(analysis_path).read_text(encoding="utf-8"))

    if template:
        with open(template, "r", encoding="utf-8") as fh:
            tmpl = jinja2.Template(fh.read())
    else:
        tmpl = jinja2.Template(_DEFAULT_REPORT_TEMPLATE)

    html = tmpl.render(
        title="ISB-1 Benchmark Report",
        results=data,
        total_cells=len(data),
        completed=sum(1 for r in data if r.get("status") == "completed"),
        failed=sum(1 for r in data if r.get("status") == "failed"),
    )

    Path(output_path).write_text(html, encoding="utf-8")
    click.echo(f"Report written to {output_path}")


_DEFAULT_REPORT_TEMPLATE = """\
<!DOCTYPE html>
<html>
<head><title>{{ title }}</title>
<style>
  body { font-family: sans-serif; margin: 2em; }
  table { border-collapse: collapse; width: 100%; }
  th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
  th { background-color: #f5f5f5; }
  .completed { color: green; }
  .failed { color: red; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>Total cells: {{ total_cells }} | Completed: {{ completed }} | Failed: {{ failed }}</p>
<table>
<tr>
  <th>GPU</th><th>Model</th><th>Workload</th><th>Mode</th><th>Quant</th>
  <th>Throughput (tok/s)</th><th>TTFT p95 (s)</th><th>TPOT p95 (s)</th>
  <th>Goodput</th><th>Status</th>
</tr>
{% for r in results %}
<tr>
  <td>{{ r.gpu }}</td>
  <td>{{ r.model }}</td>
  <td>{{ r.workload }}</td>
  <td>{{ r.mode }}</td>
  <td>{{ r.quantization }}</td>
  <td>{{ "%.1f"|format(r.generation_throughput|default(0)) }}</td>
  <td>{{ "%.4f"|format(r.ttft_p95|default(0)) }}</td>
  <td>{{ "%.4f"|format(r.tpot_p95|default(0)) }}</td>
  <td>{{ "%.2f"|format(r.goodput|default(0)) }}</td>
  <td class="{{ r.status }}">{{ r.status }}</td>
</tr>
{% endfor %}
</table>
</body>
</html>
"""


# ── quality ──────────────────────────────────────────────────────────────


@main.command()
@click.option(
    "--results-dir",
    "--input",
    required=True,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Directory containing benchmark results.",
)
@click.option(
    "--config-root",
    default=default_config_root(),
    show_default=False,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Root directory for config files. Defaults to the product-local configs/ tree.",
)
@click.option(
    "--quality-config",
    default=None,
    type=click.Path(path_type=Path),
    callback=_resolve_existing_click_path,
    help="Path to quality config YAML.",
)
def quality(results_dir: str, config_root: str, quality_config: str | None) -> None:
    """Run quality checks on benchmark outputs.

    Validates that inference outputs match reference quality expectations
    (ROUGE scores, accuracy thresholds, etc.).
    """
    results_path = Path(results_dir)

    # Find all completed runs
    manifests: list[dict] = []
    for mp in sorted(results_path.rglob("manifest.json")):
        data = json.loads(mp.read_text(encoding="utf-8"))
        if data.get("status") == "completed":
            manifests.append(data)

    if not manifests:
        click.echo("No completed runs found for quality checking.")
        return

    click.echo(f"Found {len(manifests)} completed runs for quality validation.")

    # Load quality configs
    quality_dir = Path(config_root) / "quality"
    if quality_dir.exists():
        import yaml

        quality_checks: list[dict] = []
        for qf in sorted(quality_dir.glob("*.yaml")):
            with open(qf, "r", encoding="utf-8") as fh:
                quality_checks.append(yaml.safe_load(fh) or {})
        click.echo(f"Loaded {len(quality_checks)} quality check definitions.")
    else:
        click.echo("No quality config directory found.")
        return

    click.echo("\nQuality validation requires model outputs to be captured.")
    click.echo("Run benchmarks with output capture enabled, then re-run this command.")


if __name__ == "__main__":
    main()
