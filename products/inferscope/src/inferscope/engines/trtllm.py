"""TRT-LLM v1.2+ engine adapter and config compiler."""

from __future__ import annotations

from typing import Any

from inferscope.engines.base import (
    ConfigCompiler,
    DeploymentInventory,
    EngineAdapter,
    EngineConfig,
)
from inferscope.optimization.serving_profile import ServingProfile


class TRTLLMCompiler(ConfigCompiler):
    """Compiles a ServingProfile into TRT-LLM args."""

    def engine_name(self) -> str:
        return "trtllm"

    def compile(self, profile: ServingProfile, inventory: DeploymentInventory) -> EngineConfig:
        cfg = EngineConfig(engine="trtllm")
        cfg.cli_flags["model_dir"] = profile.model

        # --- Parallelism ---
        if profile.topology.tp > 1:
            cfg.cli_flags["tp_size"] = profile.topology.tp
        if profile.topology.pp > 1:
            cfg.cli_flags["pp_size"] = profile.topology.pp

        # --- Memory and Cache ---
        cfg.cli_flags["kv_cache_type"] = "paged"
        cfg.cli_flags["max_batch_size"] = profile.scheduler.max_num_seqs
        cfg.cli_flags["max_num_tokens"] = profile.scheduler.max_num_batched_tokens

        # --- Disaggregated Serving / KV Cache Transfer ---
        if profile.topology.split_prefill_decode:
            if not inventory.has_rdma:
                cfg.warnings.append(
                    "CRITICAL: Disaggregated serving without RDMA causes severe bottleneck on TRT-LLM. "
                    "Either enable RDMA or disable prefill/decode splitting."
                )
            # Use TRT-LLM 1.1+ KV Cache Connector API
            cfg.cli_flags["enable_kv_cache_transfer"] = True

            connector = profile.topology.disagg_connector or "ucx"
            cfg.cli_flags["kv_cache_transfer_config"] = {
                "connector": connector,
                "overlap_compute": True
            }
            cfg.notes.append("Using TRT-LLM 1.1+ KV Cache Connector for disaggregated serving")

        # --- Build command string ---
        cmd_parts = ["trtllm-serve", "serve"]
        for k, v in cfg.cli_flags.items():
            if isinstance(v, bool):
                if v:
                    cmd_parts.append(f"--{k}")
            elif isinstance(v, dict):
                import json
                cmd_parts.append(f"--{k}")
                cmd_parts.append(f"'{json.dumps(v)}'")
            else:
                cmd_parts.append(f"--{k}")
                cmd_parts.append(str(v))
        cfg.command = " \\\n  ".join(cmd_parts)

        return cfg


class TRTLLMAdapter(EngineAdapter):
    """TRT-LLM engine adapter."""

    def engine_name(self) -> str:
        return "trtllm"

    async def detect_engine(self, endpoint: str) -> bool:
        return False  # Phase 5

    async def get_metrics(self, endpoint: str) -> dict[str, Any]:
        return {}

    async def get_config(self, endpoint: str) -> dict[str, Any]:
        return {}
