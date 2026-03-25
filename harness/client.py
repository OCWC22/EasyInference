"""BenchmarkClient — wraps vLLM benchmark_serving.py for ISB-1 runs."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_BACKEND = "openai-chat"
_DEFAULT_ENDPOINT = "/v1/chat/completions"
_DEFAULT_PERCENTILE_METRICS = "ttft,tpot,itl,e2el"


class BenchmarkClient:
    """Execute vLLM's ``benchmark_serving.py`` and collect results.

    Parameters
    ----------
    base_url : str
        Base URL of the vLLM server (e.g. ``http://localhost:8000``).
    model : str
        Model name as registered in the server (usually the HF model id).
    result_dir : str | Path
        Directory where JSON results are written.
    backend : str
        Backend identifier for benchmark_serving.py.
    dataset_name : str | None
        Name of the built-in dataset (e.g. ``"sharegpt"``).
    dataset_path : str | Path | None
        Path to a dataset file when not using a built-in name.
    seed : int
        Random seed for reproducibility.
    extra_args : list[str] | None
        Extra CLI args forwarded to benchmark_serving.py.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        result_dir: str | Path,
        backend: str = _DEFAULT_BACKEND,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str | Path] = None,
        seed: int = 42,
        extra_args: list[str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.result_dir = Path(result_dir)
        self.backend = backend
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.seed = seed
        self.extra_args = extra_args or []

    # ── Single-rate run ──────────────────────────────────────────────────

    def run(
        self,
        request_rate: float,
        num_prompts: int,
        *,
        goodput: Optional[str] = None,
        timeout: int = 7200,
    ) -> Path:
        """Run ``benchmark_serving.py`` at a single request rate.

        Parameters
        ----------
        request_rate : float
            Requests per second.  Use ``float("inf")`` for closed-loop.
        num_prompts : int
            Total number of prompts to send.
        goodput : str | None
            Goodput definition string (e.g. ``"ttft:2000,tpot:100"``).
        timeout : int
            Subprocess timeout in seconds.

        Returns
        -------
        Path
            Path to the JSON results file written by benchmark_serving.py.

        Raises
        ------
        RuntimeError
            If the benchmark process exits with a non-zero return code.
        """
        self.result_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",  # fallback: use benchmark_serving
        ]
        # Use the benchmark_serving script directly
        cmd = [
            sys.executable,
            "-m",
            "vllm.benchmarks.benchmark_serving",
            "--backend",
            self.backend,
            "--base-url",
            self.base_url,
            "--model",
            self.model,
            "--request-rate",
            str(request_rate),
            "--num-prompts",
            str(num_prompts),
            "--seed",
            str(self.seed),
            "--save-result",
            "--result-dir",
            str(self.result_dir),
            "--percentile-metrics",
            _DEFAULT_PERCENTILE_METRICS,
        ]

        if self.dataset_name:
            cmd.extend(["--dataset-name", self.dataset_name])
        if self.dataset_path:
            cmd.extend(["--dataset-path", str(self.dataset_path)])
        if goodput:
            cmd.extend(["--goodput", goodput])

        cmd.extend(self.extra_args)

        logger.info(
            "Running benchmark: rate=%.1f num_prompts=%d", request_rate, num_prompts
        )
        logger.debug("Command: %s", " ".join(cmd))

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(self.result_dir.parent),
        )

        if result.returncode != 0:
            logger.error("benchmark_serving stderr:\n%s", result.stderr)
            raise RuntimeError(
                f"benchmark_serving exited with code {result.returncode}:\n"
                f"{result.stderr[-2000:]}"
            )

        # Locate the most recent result file
        result_path = self._find_latest_result()
        logger.info("Benchmark results written to %s", result_path)
        return result_path

    # ── Sweep across rates ───────────────────────────────────────────────

    def run_sweep(
        self,
        rate_sweep: list[float],
        num_prompts: int,
        *,
        goodput: Optional[str] = None,
        timeout: int = 7200,
    ) -> list[Path]:
        """Run the benchmark at each rate in *rate_sweep* sequentially.

        Returns a list of result file paths, one per rate.
        """
        results: list[Path] = []
        for rate in rate_sweep:
            logger.info("Sweep rate %.1f req/s", rate)
            path = self.run(
                request_rate=rate,
                num_prompts=num_prompts,
                goodput=goodput,
                timeout=timeout,
            )
            results.append(path)
        return results

    # ── Result parsing ───────────────────────────────────────────────────

    @staticmethod
    def parse_results(result_path: str | Path) -> dict[str, Any]:
        """Parse a benchmark_serving JSON results file.

        Returns the parsed dict.  Handles both single-object and list formats.
        """
        path = Path(result_path)
        text = path.read_text(encoding="utf-8").strip()

        # benchmark_serving may write JSONL or a single JSON object
        if text.startswith("["):
            data = json.loads(text)
            return data[-1] if data else {}
        if text.startswith("{"):
            return json.loads(text)

        # JSONL fallback: return last line
        lines = [l for l in text.splitlines() if l.strip()]
        if lines:
            return json.loads(lines[-1])
        return {}

    # ── Internal helpers ─────────────────────────────────────────────────

    def _find_latest_result(self) -> Path:
        """Return the most recently modified JSON file in result_dir."""
        json_files = sorted(
            self.result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime
        )
        if not json_files:
            raise FileNotFoundError(
                f"No JSON result files found in {self.result_dir}"
            )
        return json_files[-1]

    def __repr__(self) -> str:
        return (
            f"BenchmarkClient(base_url={self.base_url!r}, model={self.model!r}, "
            f"result_dir={self.result_dir!r})"
        )
