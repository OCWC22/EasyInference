"""Model profile registry — maps model names to serving profiles.

InferScope tunes per MODEL CLASS with family-specific overrides, not per model name.
5 classes: Dense-GQA, Qwen3.5-Hybrid, Frontier-MLA-MoE, Compact-Agentic-MoE, Classical-MoE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from inferscope.optimization.serving_profile import ModelClass


@dataclass
class ModelVariant:
    """A specific model variant with serving-relevant specs."""

    name: str
    family: str
    model_class: ModelClass
    params_total_b: float
    params_active_b: float  # Same as total for dense models
    model_type: str  # dense | moe
    context_length: int
    attention_type: str  # GQA | MLA | MHA | hybrid
    kv_heads: int = 0
    head_dim: int = 128
    layers: int = 0
    experts_total: int = 0
    experts_active: int = 0
    vocab_size: int = 0
    mtp_speculative: bool = False

    # Serving recommendations
    serving: dict[str, Any] = field(default_factory=dict)

    # Memory estimates (approximate bytes per parameter)
    weight_bytes_fp16: float = 0.0  # Computed from params

    def __post_init__(self) -> None:
        if self.weight_bytes_fp16 == 0.0:
            self.weight_bytes_fp16 = self.params_total_b * 2e9  # 2 bytes per param FP16

    def weight_gb(self, precision: str = "fp16") -> float:
        """Estimated weight memory in GB."""
        multiplier = {
            "fp16": 2.0,
            "bf16": 2.0,
            "fp8": 1.0,
            "int8": 1.0,
            "fp4": 0.5,
            "nvfp4": 0.5,
            "int4": 0.5,
            "mxfp4": 0.5,
            "awq": 0.5,
            "gptq": 0.5,
        }.get(precision, 2.0)
        return self.params_total_b * multiplier

    def kv_cache_bytes_per_token(self, precision: str = "fp16") -> float:
        """KV cache bytes per token per layer."""
        dtype_bytes = {"fp16": 2.0, "bf16": 2.0, "fp8_e4m3": 1.0, "fp8": 1.0, "auto": 2.0}
        bpt = dtype_bytes.get(precision, 2.0)

        if self.attention_type == "MLA":
            # MLA compresses KV ~32x — use latent_dim instead of full dim
            latent_dim = self.serving.get("mla_latent_dim", 512)
            return 2 * latent_dim * bpt  # K + V latent vectors
        else:
            # Standard GQA: 2 * kv_heads * head_dim * bytes
            return 2 * self.kv_heads * self.head_dim * bpt

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "model_class": self.model_class.value,
            "params_total_b": self.params_total_b,
            "params_active_b": self.params_active_b,
            "type": self.model_type,
            "context_length": self.context_length,
            "attention_type": self.attention_type,
            "kv_heads": self.kv_heads,
            "head_dim": self.head_dim,
            "layers": self.layers,
            "experts_total": self.experts_total,
            "experts_active": self.experts_active,
            "mtp_speculative": self.mtp_speculative,
            "weight_gb_fp16": round(self.weight_gb("fp16"), 1),
            "weight_gb_fp8": round(self.weight_gb("fp8"), 1),
            "serving": self.serving,
        }


# =============================================================================
# Model Profiles
# =============================================================================

_MODELS: dict[str, ModelVariant] = {}


def _register(model: ModelVariant) -> ModelVariant:
    _MODELS[model.name.lower()] = model
    return model


# --- Qwen 3.5 family (Hybrid Gated DeltaNet + GQA) ---

_register(
    ModelVariant(
        name="Qwen3.5-32B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=32,
        params_active_b=32,
        model_type="dense",
        context_length=131072,
        attention_type="hybrid",
        kv_heads=8,
        head_dim=128,
        layers=64,
        serving={"vllm_flags": "--trust-remote-code", "tp_fp8": 1, "tp_bf16": 2},
    )
)

_register(
    ModelVariant(
        name="Qwen3.5-72B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=72,
        params_active_b=72,
        model_type="dense",
        context_length=131072,
        attention_type="hybrid",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"vllm_flags": "--trust-remote-code", "tp_fp8": 2, "tp_bf16": 4},
    )
)

_register(
    ModelVariant(
        name="Qwen3.5-397B-A17B",
        family="Qwen 3.5",
        model_class=ModelClass.QWEN35_HYBRID,
        params_total_b=397,
        params_active_b=17,
        model_type="moe",
        context_length=262144,
        attention_type="hybrid",
        kv_heads=2,
        head_dim=256,
        layers=60,
        experts_total=512,
        experts_active=10,
        mtp_speculative=True,
        serving={
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "vllm_flags": "--trust-remote-code",
            "sglang_speculative": "--speculative-algo NEXTN",
            "tp_fp8": 2,
            "tp_bf16": 8,
        },
    )
)

# --- DeepSeek V3 / R1 (Frontier MLA MoE) ---

_register(
    ModelVariant(
        name="DeepSeek-V3",
        family="DeepSeek V3/R1",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=671,
        params_active_b=37,
        model_type="moe",
        context_length=131072,
        attention_type="MLA",
        kv_heads=128,
        head_dim=128,
        layers=61,
        experts_total=256,
        experts_active=8,
        serving={
            "mla_latent_dim": 512,
            "compression_ratio": 32,
            "vllm_flags": "--trust-remote-code --block-size 1",
            "tp_fp8_h200": 4,
            "tp_fp8_h100": 8,
            "tp_bf16": 16,
            "ep_recommended": True,
            "nvidia_fp4": "deepseek-ai/DeepSeek-V3-0324-FP4 -tp 4 --enable-expert-parallel",
        },
    )
)

_register(
    ModelVariant(
        name="DeepSeek-R1",
        family="DeepSeek V3/R1",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=671,
        params_active_b=37,
        model_type="moe",
        context_length=131072,
        attention_type="MLA",
        kv_heads=128,
        head_dim=128,
        layers=61,
        experts_total=256,
        experts_active=8,
        serving={
            "mla_latent_dim": 512,
            "compression_ratio": 32,
            "vllm_flags": "--trust-remote-code --block-size 1",
            "additional_flags": "--enable-reasoning --reasoning-parser deepseek_r1",
            "tp_fp8_h200": 4,
            "tp_fp8_h100": 8,
            "tp_bf16": 16,
            "ep_recommended": True,
        },
    )
)

# --- DeepSeek distills (Dense GQA — NOT MLA) ---

_register(
    ModelVariant(
        name="DeepSeek-R1-Distill-Qwen-7B",
        family="DeepSeek Distills",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=7,
        params_active_b=7,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=4,
        head_dim=128,
        layers=28,
        serving={"tp_fp16": 1, "note": "Standard Qwen architecture — no special flags needed"},
    )
)

_register(
    ModelVariant(
        name="DeepSeek-R1-Distill-Llama-70B",
        family="DeepSeek Distills",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"tp_fp8_h200": 1, "tp_fp8_h100": 2, "tp_fp16_mi300x": 1},
    )
)

# --- Kimi K2 / K2.5 (Frontier MLA MoE) ---

_register(
    ModelVariant(
        name="Kimi-K2",
        family="Kimi K2/K2.5",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=1000,
        params_active_b=32,
        model_type="moe",
        context_length=128000,
        attention_type="MLA",
        kv_heads=64,
        head_dim=192,
        layers=61,
        experts_total=384,
        experts_active=8,
        vocab_size=160000,
        serving={
            "vllm_flags": "--trust-remote-code --enforce-eager",
            "tp_min": 4,
            "tp_fp8": 4,
            "tp_bf16": 8,
        },
    )
)

_register(
    ModelVariant(
        name="Kimi-K2.5",
        family="Kimi K2/K2.5",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=1000,
        params_active_b=32,
        model_type="moe",
        context_length=256000,
        attention_type="MLA",
        kv_heads=64,
        head_dim=112,  # 7168 / 64 attention heads
        layers=61,  # 61 layers including 1 dense layer
        experts_total=384,
        experts_active=8,  # + 1 shared expert
        vocab_size=160000,
        serving={
            # === HuggingFace model IDs ===
            "hf_repo": "moonshotai/Kimi-K2.5",
            "hf_repo_nvfp4": "nvidia/Kimi-K2.5-NVFP4",
            "hf_repo_amd_mxfp4": "amd/Kimi-K2.5-MXFP4",
            # === vLLM serving (primary) ===
            "vllm_flags": (
                "--trust-remote-code --enforce-eager "
                "--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 "
                "--enable-auto-tool-choice"
            ),
            "vllm_tp_bf16": 8,  # 8x H100/H200 for BF16
            "vllm_tp_fp8": 8,  # 8x H100/H200 for FP8 (recommended)
            "vllm_tp_nvfp4": 4,  # 4x B200 for NVFP4 (Blackwell only)
            "vllm_nvfp4_cmd": (
                "vllm serve nvidia/Kimi-K2.5-NVFP4 -tp 4 "
                "--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --trust-remote-code"
            ),
            # === AMD serving ===
            "amd_mxfp4_cmd": (
                "vllm serve amd/Kimi-K2.5-MXFP4 -tp 4 "
                "--mm-encoder-tp-mode data --trust-remote-code"
            ),
            "amd_gpu": "MI300X / MI325X / MI355X",
            # === SGLang serving ===
            "sglang_cmd": (
                "sglang serve --model-path moonshotai/Kimi-K2.5 --tp 8 "
                "--trust-remote-code --tool-call-parser kimi_k2 --reasoning-parser kimi_k2"
            ),
            # === Speculative decoding ===
            "eagle3_speculative": '{"model": "lightseekorg/kimi-k2.5-eagle3", "method": "eagle3"}',
            # === KV cache ===
            "kv_cache_dtype": "fp8",  # --kv-cache-dtype fp8 halves KV memory
            "mla_latent_dim": 512,  # MLA compresses KV to latent dim
            "kv_compression_ratio": 32,  # MLA provides ~32x KV compression
            # === Disaggregated prefill ===
            "disagg_recommended": True,
            "disagg_note": "FP8 + DCP + Triton MLA is best for max throughput with long context",
            # === Generation defaults ===
            "temperature_thinking": 1.0,
            "temperature_instant": 0.6,
            "top_p": 0.95,
            # === Vision ===
            "vision_encoder": "MoonViT (400M params)",
            "mm_encoder_tp_mode": "data",  # --mm-encoder-tp-mode data (small encoder, no TP gain)
        },
    )
)

# --- GLM family (Zhipu AI / THUDM) ---

_register(
    ModelVariant(
        name="GLM-4-9B",
        family="GLM",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=9,
        params_active_b=9,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=2,
        head_dim=128,
        layers=40,
        vocab_size=151552,
        serving={
            "vllm_flags": "--trust-remote-code",
            "tp_fp16": 1,
            "tp_fp8": 1,
        },
    )
)

_register(
    ModelVariant(
        name="GLM-4.7",
        family="GLM",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=355,
        params_active_b=50,
        model_type="moe",
        context_length=1048576,
        attention_type="GQA",
        kv_heads=8,
        layers=60,
        experts_total=160,
        experts_active=8,
        serving={
            "vllm_flags": "--trust-remote-code",
            "mtp_speculative": "--speculative-config.method mtp --speculative-config.num_speculative_tokens 1",
            "mtp_acceptance_rate": ">90%",
            "tp_fp8": 4,
            "tp_bf16": 8,
        },
    )
)

_register(
    ModelVariant(
        name="GLM-5",
        family="GLM",
        model_class=ModelClass.FRONTIER_MLA_MOE,
        params_total_b=744,
        params_active_b=40,
        model_type="moe",
        context_length=202752,  # max_position_embeddings from config.json
        attention_type="MLA",  # Multi-Head Latent Attention (DeepSeek-style)
        kv_heads=64,  # num_key_value_heads=64 (before MLA compression)
        head_dim=256,  # qk_head_dim=256, v_head_dim=256
        layers=78,  # num_hidden_layers=78 (3 dense + 75 MoE)
        experts_total=256,
        experts_active=8,  # + 1 shared expert
        vocab_size=154880,
        serving={
            # === HuggingFace model IDs ===
            "hf_repo": "zai-org/GLM-5",
            "hf_repo_fp8": "zai-org/GLM-5-FP8",
            "hf_repo_nvfp4": "nvidia/GLM-5-NVFP4",
            # === vLLM serving (primary) ===
            "vllm_flags": (
                "--trust-remote-code "
                "--tool-call-parser glm47 --reasoning-parser glm45 "
                "--enable-auto-tool-choice"
            ),
            "vllm_fp8_cmd": (
                "vllm serve zai-org/GLM-5-FP8 -tp 8 "
                "--gpu-memory-utilization 0.85 "
                "--speculative-config.method mtp --speculative-config.num_speculative_tokens 1 "
                "--tool-call-parser glm47 --reasoning-parser glm45 --enable-auto-tool-choice"
            ),
            "vllm_tp_fp8": 8,  # 8x H100/H200 for FP8
            "vllm_tp_bf16": 16,  # 16x H100 for BF16 (or 8x H200)
            # === SGLang serving ===
            "sglang_cmd": (
                "python3 -m sglang.launch_server --model-path zai-org/GLM-5-FP8 --tp-size 8 "
                "--tool-call-parser glm47 --reasoning-parser glm45 "
                "--speculative-algorithm EAGLE --speculative-num-steps 3 "
                "--speculative-eagle-topk 1 --speculative-num-draft-tokens 4 "
                "--mem-fraction-static 0.85"
            ),
            # === NVFP4 serving (Blackwell only) ===
            "nvfp4_cmd": (
                "python3 -m sglang.launch_server --model nvidia/GLM-5-NVFP4 -tp 8 "
                "--quantization modelopt_fp4 --chunked-prefill-size 131072 "
                "--mem-fraction-static 0.80 "
                "--tool-call-parser glm47 --reasoning-parser glm45 --trust-remote-code"
            ),
            "nvfp4_gpu": "B200 / B300 (Blackwell only)",
            # === Speculative decoding ===
            "mtp_speculative": (
                "--speculative-config.method mtp --speculative-config.num_speculative_tokens 1"
            ),
            "num_nextn_predict_layers": 1,
            # === MLA / KV cache architecture ===
            "mla_q_lora_rank": 2048,
            "mla_kv_lora_rank": 512,  # KV cache compressed to 512-dim latent
            "mla_qk_nope_head_dim": 192,
            "mla_qk_rope_head_dim": 64,
            "kv_cache_dtype": "fp8",  # --kv-cache-dtype fp8 recommended
            # === Sparse attention ===
            "sparse_attention": "DeepSeek Sparse Attention (DSA)",
            "sparse_attention_note": "Reduces deployment cost while preserving long-context capacity",
            # === Disaggregated prefill ===
            "disagg_recommended": True,
            "first_k_dense_replace": 3,  # First 3 layers are dense (no MoE)
            "ep_recommended": True,
            # === Generation defaults ===
            "temperature": 0.7,
            "top_p": 0.95,
            "gpu_memory_utilization": 0.85,
        },
    )
)

# --- MiniMax (Compact Agentic MoE) ---

_register(
    ModelVariant(
        name="MiniMax-M2.5",
        family="MiniMax",
        model_class=ModelClass.COMPACT_AGENTIC_MOE,
        params_total_b=230,
        params_active_b=10,
        model_type="moe",
        context_length=1048576,
        attention_type="GQA",
        kv_heads=0,  # Under-documented
        serving={
            "vllm_flags": (
                "--trust-remote-code --tool-call-parser minimax_m2 --reasoning-parser minimax_m2_append_think"
            ),
            "tp_warning": "Pure TP=8 NOT supported — max pure TP is 4, use EP for 8-GPU",
            "tp_with_ep": "-tp 4 --enable-expert-parallel",
        },
    )
)

# --- Mixtral (Classical MoE) ---

_register(
    ModelVariant(
        name="Mixtral-8x7B",
        family="Mixtral",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=46.7,
        params_active_b=13,
        model_type="moe",
        context_length=32768,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=32,
        experts_total=8,
        experts_active=2,
        serving={"tp_fp8_h100": 1, "tp_fp16_a100": 2},
    )
)

_register(
    ModelVariant(
        name="Mixtral-8x22B",
        family="Mixtral",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=141,
        params_active_b=39,
        model_type="moe",
        context_length=65536,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=56,
        experts_total=8,
        experts_active=2,
        serving={"tp_fp8": 2, "tp_fp16": 4},
    )
)

# --- Qwen 2.5 family (Dense GQA) ---

_register(
    ModelVariant(
        name="Qwen2.5-7B-Instruct",
        family="Qwen 2.5",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=7,
        params_active_b=7,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=4,
        head_dim=128,
        layers=28,
        vocab_size=152064,
        serving={"tp_fp16": 1, "tp_fp8": 1},
    )
)

_register(
    ModelVariant(
        name="Qwen2.5-32B-Instruct",
        family="Qwen 2.5",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=32,
        params_active_b=32,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=64,
        vocab_size=152064,
        serving={"tp_fp8": 1, "tp_bf16": 2},
    )
)

_register(
    ModelVariant(
        name="Qwen2.5-72B-Instruct",
        family="Qwen 2.5",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=72,
        params_active_b=72,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        vocab_size=152064,
        serving={"tp_fp8": 2, "tp_bf16": 4},
    )
)

# --- Llama 3/3.3/4 family (Dense GQA) ---

_register(
    ModelVariant(
        name="Llama-3-8B",
        family="Llama 3",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=8,
        params_active_b=8,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=32,
        serving={"tp_fp16": 1, "tp_fp8": 1},
    )
)

_register(
    ModelVariant(
        name="Llama-3-70B",
        family="Llama 3",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"tp_fp8_h200": 1, "tp_fp8_h100": 2, "tp_fp16": 4},
    )
)

_register(
    ModelVariant(
        name="Llama-3.3-70B-Instruct",
        family="Llama 3.3",
        model_class=ModelClass.DENSE_GQA,
        params_total_b=70,
        params_active_b=70,
        model_type="dense",
        context_length=131072,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=80,
        serving={"tp_fp8_h200": 1, "tp_fp8_h100": 2, "tp_fp16": 4},
    )
)

_register(
    ModelVariant(
        name="Llama-4-Maverick",
        family="Llama 4",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=400,
        params_active_b=17,
        model_type="moe",
        context_length=1048576,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=48,
        experts_total=128,
        experts_active=1,
        serving={"tp_fp8": 4, "tp_bf16": 8, "note": "Natively multimodal, interleaved attention"},
    )
)

_register(
    ModelVariant(
        name="Llama-4-Scout",
        family="Llama 4",
        model_class=ModelClass.CLASSICAL_MOE,
        params_total_b=109,
        params_active_b=17,
        model_type="moe",
        context_length=524288,
        attention_type="GQA",
        kv_heads=8,
        head_dim=128,
        layers=48,
        experts_total=16,
        experts_active=1,
        serving={"tp_fp8": 1, "tp_bf16": 2, "note": "Natively multimodal, interleaved attention"},
    )
)


# =============================================================================
# Lookup functions
# =============================================================================


def get_model_variant(name: str) -> ModelVariant | None:
    """Look up a model by name (case-insensitive, flexible matching)."""
    key = name.lower().strip()

    # Direct match
    if key in _MODELS:
        return _MODELS[key]

    # Fuzzy match: try removing vendor prefixes, hyphens, etc.
    normalized = key.replace("/", "-").replace("_", "-")
    for model_key, model in _MODELS.items():
        if normalized in model_key or model_key in normalized:
            return model
        # Check against the full HuggingFace-style name
        if normalized.endswith(model_key):
            return model

    return None


def list_models() -> list[str]:
    """List all known model names."""
    return sorted(set(m.name for m in _MODELS.values()))


def get_models_by_class(model_class: ModelClass) -> list[ModelVariant]:
    """Get all models in a given class."""
    return [m for m in _MODELS.values() if m.model_class == model_class]
