"""BenchmarkRunner — single-cell executor for ISB-1 benchmarks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from harness.client import BenchmarkClient
from harness.config_validator import ConfigValidator
from harness.engine_metrics import EngineMetricsCollector
from harness.lockfile import LockfileGenerator
from harness.manifest import RunManifest
from harness.server import VLLMServer
from harness.telemetry import TelemetryCollector
from harness.warmup import WarmupValidator

logger = logging.getLogger(__name__)


@dataclass
class CellConfig:
    """All parameters needed to execute a single benchmark cell."""

    # Hardware
    gpu: str = ""
    gpu_count: int = 1

    # Model
    model: str = ""
    model_hf_id: str = ""
    model_revision: str = ""

    # Benchmark parameters
    workload: str = ""
    mode: str = ""
    quantization: str = "fp8"
    topology: str = "tp1"
    kv_cache_dtype: str = "auto"

    # Trial
    trial_number: int = 1

    # Server
    port: int = 8000
    startup_timeout: int = 600
    vllm_extra_args: list[str] = field(default_factory=list)

    # Benchmark
    num_prompts: int = 1000
    rate_sweep: list[float] = field(default_factory=lambda: [1.0])
    seed: int = 42
    dataset_name: Optional[str] = None
    dataset_path: Optional[str | Path] = None
    goodput: Optional[str] = None

    # Warmup
    warmup_requests: int = 100
    warmup_seconds: float = 60.0
    warmup_max_extensions: int = 3
    warmup_variance_threshold: float = 0.20

    # Measurement
    measurement_duration_seconds: float = 600.0
    cooldown_seconds: float = 30.0

    # Paths
    output_dir: str | Path = "results"
    config_root: str | Path = "configs"

    # Config file paths (for lockfile hashing)
    config_paths: list[str | Path] = field(default_factory=list)


@dataclass
class RunResult:
    """Paths and status from a completed benchmark cell."""

    run_id: str = ""
    status: str = "completed"
    error_message: Optional[str] = None

    manifest_path: Optional[Path] = None
    lockfile_path: Optional[Path] = None
    benchmark_results: list[Path] = field(default_factory=list)
    telemetry_path: Optional[Path] = None
    engine_metrics_path: Optional[Path] = None
    server_log_path: Optional[Path] = None

    warmup_stable: bool = False
    warmup_extensions: int = 0
    startup_time_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_id": self.run_id,
            "status": self.status,
            "error_message": self.error_message,
            "warmup_stable": self.warmup_stable,
            "warmup_extensions": self.warmup_extensions,
            "startup_time_seconds": self.startup_time_seconds,
        }
        for key in (
            "manifest_path",
            "lockfile_path",
            "telemetry_path",
            "engine_metrics_path",
            "server_log_path",
        ):
            val = getattr(self, key)
            d[key] = str(val) if val else None
        d["benchmark_results"] = [str(p) for p in self.benchmark_results]
        return d


class BenchmarkRunner:
    """Orchestrate a single ISB-1 benchmark cell execution.

    Lifecycle
    ---------
    1. Validate configuration
    2. Generate manifest and run ID
    3. Start vLLM server
    4. Health check (implicit in server.start())
    5. Start telemetry and engine metrics collection
    6. Run warmup validation
    7. Execute benchmark
    8. Stop telemetry and engine metrics
    9. Stop server
    10. Save results, manifest, and lockfile
    """

    def __init__(self, cell: CellConfig) -> None:
        self.cell = cell

    # ── Run ID generation ────────────────────────────────────────────────

    @staticmethod
    def _generate_run_id(cell: CellConfig) -> str:
        """Format: isb1-{date}-{gpu}-{model}-{workload}-{mode}-{quant}-{trial:03d}"""
        date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        parts = [
            "isb1",
            date_str,
            cell.gpu,
            cell.model,
            cell.workload,
            cell.mode,
            cell.quantization,
            f"{cell.trial_number:03d}",
        ]
        return "-".join(parts)

    # ── Config hash ──────────────────────────────────────────────────────

    @staticmethod
    def _compute_config_hash(cell: CellConfig) -> str:
        """SHA-256 of the cell's essential parameters for integrity."""
        canonical = json.dumps(
            {
                "gpu": cell.gpu,
                "gpu_count": cell.gpu_count,
                "model": cell.model,
                "model_hf_id": cell.model_hf_id,
                "workload": cell.workload,
                "mode": cell.mode,
                "quantization": cell.quantization,
                "topology": cell.topology,
                "kv_cache_dtype": cell.kv_cache_dtype,
                "num_prompts": cell.num_prompts,
                "seed": cell.seed,
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()

    # ── vLLM args builder ────────────────────────────────────────────────

    def _build_vllm_args(self) -> list[str]:
        """Construct extra CLI args for vllm serve."""
        args: list[str] = []
        cell = self.cell

        # Tensor parallelism from topology
        if cell.topology.startswith("tp"):
            tp = cell.topology[2:]
            if tp.isdigit():
                args.extend(["--tensor-parallel-size", tp])

        # Quantization
        if cell.quantization and cell.quantization not in ("bf16", "fp16"):
            args.extend(["--quantization", cell.quantization])

        # Dtype for bf16/fp16
        if cell.quantization in ("bf16", "fp16"):
            args.extend(["--dtype", cell.quantization])

        # KV cache dtype
        if cell.kv_cache_dtype and cell.kv_cache_dtype != "auto":
            args.extend(["--kv-cache-dtype", cell.kv_cache_dtype])

        # GPU memory utilisation (leave headroom)
        args.extend(["--gpu-memory-utilization", "0.90"])

        # Extra user args
        args.extend(cell.vllm_extra_args)

        return args

    # ── Main execution ───────────────────────────────────────────────────

    def run(self) -> RunResult:
        """Execute the full benchmark lifecycle. Returns a RunResult."""
        cell = self.cell
        run_id = self._generate_run_id(cell)
        run_dir = Path(cell.output_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        result = RunResult(run_id=run_id)
        manifest = RunManifest(
            run_id=run_id,
            gpu=cell.gpu,
            gpu_count=cell.gpu_count,
            model=cell.model,
            model_hf_id=cell.model_hf_id,
            model_revision=cell.model_revision,
            workload=cell.workload,
            mode=cell.mode,
            quantization=cell.quantization,
            topology=cell.topology,
            kv_cache_dtype=cell.kv_cache_dtype,
            config_hash=self._compute_config_hash(cell),
            trial_number=cell.trial_number,
            total_requests=cell.num_prompts,
            warmup_requests=cell.warmup_requests,
        )
        manifest.stamp_start()

        server: Optional[VLLMServer] = None
        telemetry: Optional[TelemetryCollector] = None
        engine_metrics: Optional[EngineMetricsCollector] = None

        try:
            # Step 1: Validate configuration
            logger.info("[%s] Validating configuration", run_id)
            validator = ConfigValidator(cell.config_root)
            vr = validator.check_memory_fit(
                cell.gpu, cell.model, cell.quantization, cell.gpu_count
            )
            if not vr.ok:
                raise RuntimeError(f"Config validation failed:\n{vr.summary()}")

            # Step 2: Start server
            logger.info("[%s] Starting vLLM server", run_id)
            vllm_args = self._build_vllm_args()
            server = VLLMServer(
                model=cell.model_hf_id,
                port=cell.port,
                extra_args=vllm_args,
                log_dir=run_dir / "logs",
                startup_timeout=cell.startup_timeout,
            )
            server.start()
            result.startup_time_seconds = server.startup_time_seconds or 0.0
            result.server_log_path = run_dir / "logs" / "vllm_server.log"

            # Step 3: Start telemetry
            logger.info("[%s] Starting telemetry collection", run_id)
            telemetry = TelemetryCollector(
                output_path=run_dir / "telemetry.csv",
            )
            telemetry.start()
            result.telemetry_path = run_dir / "telemetry.csv"

            # Step 4: Start engine metrics
            logger.info("[%s] Starting engine metrics collection", run_id)
            engine_metrics = EngineMetricsCollector(
                metrics_url=server.get_metrics_url(),
                output_path=run_dir / "engine_metrics.jsonl",
            )
            engine_metrics.start()
            result.engine_metrics_path = run_dir / "engine_metrics.jsonl"

            # Step 5: Run benchmark (with rate sweep)
            logger.info("[%s] Running benchmark", run_id)
            client = BenchmarkClient(
                base_url=server.base_url,
                model=cell.model_hf_id,
                result_dir=run_dir / "benchmark",
                dataset_name=cell.dataset_name,
                dataset_path=cell.dataset_path,
                seed=cell.seed,
            )

            benchmark_paths = client.run_sweep(
                rate_sweep=cell.rate_sweep,
                num_prompts=cell.num_prompts,
                goodput=cell.goodput,
            )
            result.benchmark_results = benchmark_paths

            # Step 6: Warmup validation (on the last result set)
            logger.info("[%s] Validating warmup", run_id)
            if benchmark_paths:
                raw_data = client.parse_results(benchmark_paths[-1])
                per_request = raw_data.get("per_request", [])
                if per_request:
                    warmup = WarmupValidator(
                        warmup_requests=cell.warmup_requests,
                        warmup_seconds=cell.warmup_seconds,
                        max_extensions=cell.warmup_max_extensions,
                        variance_threshold=cell.warmup_variance_threshold,
                    )
                    wr = warmup.validate(per_request)
                    manifest.warmup_stable = wr.is_stable
                    result.warmup_stable = wr.is_stable
                    result.warmup_extensions = wr.warmup_extensions

            manifest.status = "completed"
            result.status = "completed"

        except Exception as exc:
            logger.exception("[%s] Benchmark failed: %s", run_id, exc)
            manifest.status = "failed"
            manifest.error_message = str(exc)
            result.status = "failed"
            result.error_message = str(exc)

        finally:
            # Step 7: Stop telemetry
            if telemetry is not None:
                try:
                    telemetry.stop()
                except Exception as exc:
                    logger.warning("Failed to stop telemetry: %s", exc)

            # Step 8: Stop engine metrics
            if engine_metrics is not None:
                try:
                    engine_metrics.stop()
                except Exception as exc:
                    logger.warning("Failed to stop engine metrics: %s", exc)

            # Step 9: Stop server
            if server is not None:
                try:
                    server.stop()
                except Exception as exc:
                    logger.warning("Failed to stop server: %s", exc)

            # Cooldown
            if cell.cooldown_seconds > 0:
                logger.info(
                    "[%s] Cooldown for %.0fs", run_id, cell.cooldown_seconds
                )
                time.sleep(cell.cooldown_seconds)

            # Step 10: Save manifest and lockfile
            manifest.stamp_end()
            elapsed = 0.0
            if manifest.timestamp_start and manifest.timestamp_end:
                try:
                    t_start = datetime.fromisoformat(manifest.timestamp_start)
                    t_end = datetime.fromisoformat(manifest.timestamp_end)
                    elapsed = (t_end - t_start).total_seconds()
                except ValueError:
                    pass
            manifest.duration_seconds = elapsed

            manifest_path = run_dir / "manifest.json"
            manifest.save(manifest_path)
            result.manifest_path = manifest_path

            # Lockfile
            lockgen = LockfileGenerator()
            engine_args = {
                "model": cell.model_hf_id,
                "port": cell.port,
                "extra_args": self._build_vllm_args(),
            }
            lockgen.generate(
                model_hf_id=cell.model_hf_id,
                engine_args=engine_args,
                config_paths=cell.config_paths,
                random_seeds={"benchmark": cell.seed},
            )
            lockfile_path = run_dir / "lockfile.json"
            lockgen.save(lockfile_path)
            result.lockfile_path = lockfile_path

            # Save run result summary
            summary_path = run_dir / "run_result.json"
            summary_path.write_text(
                json.dumps(result.to_dict(), indent=2, default=str) + "\n",
                encoding="utf-8",
            )

        logger.info(
            "[%s] Run complete: status=%s, dir=%s",
            run_id,
            result.status,
            run_dir,
        )
        return result

    def __repr__(self) -> str:
        return (
            f"BenchmarkRunner(model={self.cell.model!r}, gpu={self.cell.gpu!r}, "
            f"workload={self.cell.workload!r})"
        )
