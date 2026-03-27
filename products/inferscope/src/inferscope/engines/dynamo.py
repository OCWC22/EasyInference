"""NVIDIA Dynamo v1.0+ engine adapter and config compiler.

Dynamo is a datacenter-scale distributed inference serving framework.
Production-ready since March 16, 2026 (GTC announcement). Deployed at
AWS, Azure, Google Cloud, Oracle, Baseten, Fireworks, Cursor, Perplexity.

Key components:
- SLO Planner: capacity monitoring + scaling decisions
- KV-aware Router: routes requests to workers with matching KV cache
- NIXL: low-latency GPU-to-GPU KV cache transfer library
- Grove: hierarchical KV block manager (GPU -> CPU -> NVMe -> S3)
- Multi-backend: supports vLLM, SGLang, and TRT-LLM as workers

Dynamo uses declarative YAML topology configs, not CLI flags.
"""

from __future__ import annotations

from typing import Any

import httpx

from inferscope.endpoint_auth import EndpointAuthConfig, build_auth_headers
from inferscope.engines.base import (
    ConfigCompiler,
    DeploymentInventory,
    EngineAdapter,
    EngineConfig,
)
from inferscope.logging import get_logger
from inferscope.optimization.serving_profile import ServingProfile

_adapter_log = get_logger(component="dynamo_adapter")


class DynamoCompiler(ConfigCompiler):
    """Compiles a ServingProfile into a Dynamo deployment config.

    Dynamo uses a declarative YAML topology config rather than CLI flags.
    The compiler generates the YAML structure as a nested dict in cli_flags,
    plus environment variables and deployment notes.
    """

    def engine_name(self) -> str:
        return "dynamo"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="dynamo")
        cfg.support_tier = "supported"
        cfg.support_reason = (
            "Dynamo 1.0 is production-ready for disaggregated NVIDIA deployments "
            "(GTC 2026, deployed at AWS/Azure/GCP/Oracle + Baseten, Fireworks, Cursor, Perplexity)."
        )

        if not inventory.gpu_arch.startswith("sm_"):
            cfg.support_tier = "unsupported"
            cfg.support_reason = "Dynamo requires NVIDIA GPUs (SM architecture)."
            cfg.warnings.append(cfg.support_reason)
            return cfg

        # --- Topology ---
        topology = {
            "model": profile.model,
            "tensor_parallel_size": profile.topology.tp,
            "pipeline_parallel_size": profile.topology.pp,
        }
        if profile.topology.dp > 1:
            topology["data_parallel_size"] = profile.topology.dp
        if profile.topology.ep > 1:
            topology["expert_parallel_size"] = profile.topology.ep

        # --- Backend engine ---
        # Dynamo can use vLLM, SGLang, or TRT-LLM as worker backends
        backend = "vllm"  # default
        if profile.engine and profile.engine.value in ("vllm", "sglang", "trtllm"):
            backend = profile.engine.value
        topology["backend"] = backend

        # --- Disaggregated serving ---
        disagg: dict[str, Any] = {}
        if profile.topology.split_prefill_decode:
            disagg["enabled"] = True
            disagg["prefill_workers"] = max(1, profile.topology.tp)
            disagg["decode_workers"] = max(1, profile.topology.tp)
            disagg["kv_transfer"] = "nixl"

            if inventory.has_rdma:
                disagg["transport"] = "rdma"
                cfg.notes.append("RDMA enabled for KV transfer — optimal disagg performance.")
            else:
                disagg["transport"] = "tcp"
                cfg.warnings.append(
                    "Disaggregated serving without RDMA: expect 20-30% degradation in "
                    "KV transfer latency. Enable RoCE or InfiniBand for production."
                )

            if profile.topology.disagg_connector:
                disagg["connector"] = profile.topology.disagg_connector

            # NVLink5 advantage on Blackwell
            if inventory.gpu_arch in ("sm_100", "sm_103"):
                bw = inventory.interconnect_bandwidth_gb_s
                cfg.notes.append(
                    f"Blackwell NVLink5 @ {bw:.0f} GB/s enables compressed KV transfer "
                    "via nvCOMP decompression engine during P/D handoff."
                )

        # --- KV cache management (Grove) ---
        grove: dict[str, Any] = {
            "gpu_memory_utilization": profile.cache.gpu_memory_utilization,
        }
        if profile.cache.kv_tiering == "gpu_cpu":
            grove["tiers"] = ["gpu_hbm", "cpu_dram"]
            if inventory.has_grace:
                grove["tiers"] = ["gpu_hbm", "grace_lpddr5x", "cpu_dram"]
                grove["c2c_bandwidth_gb_s"] = inventory.c2c_bandwidth_gb_s
                cfg.notes.append(
                    f"Grace LPDDR5X ({inventory.grace_memory_gb:.0f}GB @ "
                    f"{inventory.grace_memory_bandwidth_gb_s:.0f} GB/s) as KV overflow "
                    f"via NVLink-C2C @ {inventory.c2c_bandwidth_gb_s:.0f} GB/s."
                )
        elif profile.cache.kv_tiering == "gpu_cpu_ssd":
            grove["tiers"] = ["gpu_hbm", "cpu_dram", "nvme_ssd"]
        else:
            grove["tiers"] = ["gpu_hbm"]

        if profile.cache.prefix_cache:
            grove["prefix_caching"] = True

        # --- SLO Planner ---
        slo_planner = {}
        if profile.objective.ttft_p95_ms > 0:
            slo_planner["ttft_p95_ms"] = profile.objective.ttft_p95_ms
        if profile.objective.itl_p95_ms > 0:
            slo_planner["itl_p95_ms"] = profile.objective.itl_p95_ms
        if profile.objective.throughput_min_tps > 0:
            slo_planner["min_throughput_tps"] = profile.objective.throughput_min_tps

        # --- Router ---
        router = {"type": "kv_aware"}
        if profile.cache.session_affinity:
            router["sticky_sessions"] = True

        # --- Precision ---
        precision = {}
        if profile.precision.weights == "fp8":
            precision["quantization"] = "fp8"
        elif profile.precision.weights == "fp4":
            if inventory.gpu_arch in ("sm_100", "sm_103"):
                precision["quantization"] = "nvfp4"
            else:
                cfg.warnings.append(f"NVFP4 requires Blackwell, got {inventory.gpu_arch}")
                precision["quantization"] = "fp8"
        elif profile.precision.weights in ("awq", "gptq"):
            precision["quantization"] = profile.precision.weights
        if profile.precision.kv_cache and profile.precision.kv_cache != "auto":
            precision["kv_cache_dtype"] = profile.precision.kv_cache

        # --- Scheduler ---
        scheduler = {
            "max_num_batched_tokens": profile.scheduler.batched_token_budget,
            "max_num_seqs": profile.scheduler.max_num_seqs,
        }
        if profile.scheduler.chunked_prefill:
            scheduler["enable_chunked_prefill"] = True
        if profile.scheduler.enable_moe_overlap:
            scheduler["enable_moe_overlap"] = True

        # --- Speculation ---
        if profile.speculation and profile.speculation.mode != "off":
            scheduler["speculative_config"] = {
                "method": profile.speculation.method,
                "num_speculative_tokens": profile.speculation.num_speculative_tokens,
            }

        # --- EPLB ---
        if profile.topology.enable_eplb:
            topology["enable_eplb"] = True
            cfg.notes.append("EPLB enabled for dynamic MoE expert rebalancing across ranks.")

        # --- Assemble config YAML structure ---
        dynamo_config = {
            "topology": topology,
            "precision": precision,
            "scheduler": scheduler,
            "grove": grove,
            "router": router,
        }
        if disagg:
            dynamo_config["disaggregated"] = disagg
        if slo_planner:
            dynamo_config["slo_planner"] = slo_planner

        cfg.cli_flags = dynamo_config

        # --- Env vars ---
        if inventory.gpu_arch in ("sm_100", "sm_103") and inventory.has_decompression_engine:
            cfg.env_vars["DYNAMO_ENABLE_NVCOMP"] = "1"
            cfg.notes.append("nvCOMP decompression engine enabled for compressed KV transfer.")

        # --- Command ---
        cfg.command = (
            f"dynamo serve \\\n"
            f"  --config dynamo-config.yaml \\\n"
            f"  --model {profile.model}"
        )

        # --- Platform-specific notes ---
        if inventory.gpu_arch == "sm_100":
            cfg.notes.append(
                "Blackwell B200: Dynamo + NIXL achieves up to 30x token throughput "
                "for DeepSeek-R1 on GB200 NVL72 vs single-node baseline."
            )
        elif inventory.gpu_arch == "sm_103":
            cfg.notes.append(
                "Blackwell B300: inference-optimized die with accelerated softmax. "
                "Dynamo enables full NVLink5 mesh utilization for P/D disaggregation."
            )
        elif inventory.gpu_arch == "sm_90a":
            cfg.notes.append(
                "Hopper H100/H200: Dynamo provides KV-aware routing and NIXL transfer "
                "over NVLink4 @ 450 GB/s. Recommended for multi-node disaggregated deployments."
            )

        return cfg


class DynamoAdapter(EngineAdapter):
    """Connects to a running Dynamo deployment.

    Dynamo exposes metrics through its router/gateway component.
    Detection checks for Dynamo-specific metrics or the Dynamo health endpoint.
    """

    def engine_name(self) -> str:
        return "dynamo"

    async def detect_engine(self, endpoint: str) -> bool:
        """Detect Dynamo by checking for dynamo-specific metrics or health response."""
        try:
            url = self._validate_endpoint(endpoint)
            async with httpx.AsyncClient(timeout=5.0) as client:
                # Check /metrics for dynamo-specific prefixes
                resp = await client.get(f"{url}/metrics")
                if "dynamo_" in resp.text or "dynamo:" in resp.text:
                    return True

                # Check /health for Dynamo-specific response
                try:
                    health = await client.get(f"{url}/health")
                    if health.status_code == 200:
                        body = health.text.lower()
                        if "dynamo" in body:
                            return True
                except Exception:  # noqa: S110
                    pass

                # Check for Dynamo router headers
                try:
                    models = await client.get(f"{url}/v1/models")
                    if "x-dynamo" in {k.lower() for k in models.headers}:
                        return True
                except Exception:  # noqa: S110
                    pass

        except Exception:  # noqa: S110
            pass
        return False

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        """Scrape Prometheus metrics from Dynamo's gateway/router.

        Dynamo exposes both its own metrics (dynamo_*) and backend engine
        metrics (vllm:*, sglang:*) through the gateway.
        """
        url = self._validate_endpoint(endpoint)
        metrics: dict[str, Any] = {}
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(f"{url}/metrics")
                for line in resp.text.splitlines():
                    if line.startswith("#"):
                        continue
                    # Capture dynamo-specific and backend metrics
                    if any(prefix in line for prefix in ("dynamo_", "dynamo:", "vllm:", "sglang:")):
                        parts = line.split()
                        if len(parts) >= 2:
                            name = parts[0].split("{")[0]
                            try:
                                metrics[name] = float(parts[-1])
                            except ValueError:
                                metrics[name] = parts[-1]
        except Exception:  # noqa: S110
            _adapter_log.warning("dynamo_metrics_scrape_failed", endpoint=endpoint)
        return metrics

    async def get_config(
        self,
        endpoint: str,
        *,
        allow_private: bool = True,
        auth: EndpointAuthConfig | None = None,
    ) -> dict[str, Any]:
        """Get deployment info from Dynamo's management API."""
        try:
            url = self._validate_endpoint(endpoint, allow_private=allow_private)
            async with httpx.AsyncClient(timeout=10.0) as client:
                # Try /v1/models (standard OpenAI-compatible)
                resp = await client.get(f"{url}/v1/models", headers=build_auth_headers(auth))
                config: dict[str, Any] = resp.json()

                # Try Dynamo-specific status endpoint
                try:
                    status = await client.get(f"{url}/v1/status", headers=build_auth_headers(auth))
                    if status.status_code == 200:
                        config["dynamo_status"] = status.json()
                except Exception:  # noqa: S110
                    pass

                return config
        except Exception:  # noqa: S110
            _adapter_log.warning("dynamo_config_fetch_failed", endpoint=endpoint)
            return {}
