"""Tests for stack plan generation reliability with production models and disaggregation."""

from __future__ import annotations

import pytest

from inferscope.benchmarks import (
    BenchmarkConnectionError,
    assess_benchmark_support,
    build_benchmark_stack_plan,
    list_builtin_experiments,
    list_builtin_workloads,
    load_experiment,
    load_workload,
    materialize_workload,
)
from inferscope.models.registry import get_model_variant, list_models


# ---------------------------------------------------------------------------
# Model registry reliability
# ---------------------------------------------------------------------------


class TestModelRegistry:
    """Verify all priority models resolve correctly."""

    @pytest.mark.parametrize(
        "model_name",
        [
            "Kimi-K2",
            "Kimi-K2.5",
            "GLM-4-9B",
            "GLM-4.7",
            "GLM-5",
            "Qwen3.5-32B",
            "DeepSeek-R1",
            "Llama-3.3-70B-Instruct",
        ],
    )
    def test_model_resolves(self, model_name: str) -> None:
        variant = get_model_variant(model_name)
        assert variant is not None, f"Model '{model_name}' not found in registry"
        assert variant.params_total_b > 0
        assert variant.context_length > 0

    @pytest.mark.parametrize(
        "model_name,expected_type",
        [
            ("Kimi-K2", "moe"),
            ("GLM-5", "moe"),
            ("GLM-4-9B", "dense"),
            ("Kimi-K2.5", "moe"),
            ("GLM-4.7", "moe"),
        ],
    )
    def test_model_type(self, model_name: str, expected_type: str) -> None:
        variant = get_model_variant(model_name)
        assert variant is not None
        assert variant.model_type == expected_type

    @pytest.mark.parametrize(
        "model_name,expected_attention",
        [
            ("Kimi-K2", "MLA"),
            ("Kimi-K2.5", "MLA"),
            ("GLM-5", "GQA"),
            ("GLM-4-9B", "GQA"),
        ],
    )
    def test_model_attention_type(self, model_name: str, expected_attention: str) -> None:
        variant = get_model_variant(model_name)
        assert variant is not None
        assert variant.attention_type == expected_attention

    def test_fuzzy_match_kimi(self) -> None:
        """Kimi K2 should match various name formats."""
        assert get_model_variant("kimi-k2") is not None
        assert get_model_variant("Kimi-K2") is not None

    def test_fuzzy_match_glm(self) -> None:
        """GLM models should match various name formats."""
        assert get_model_variant("glm-5") is not None
        assert get_model_variant("GLM-5") is not None
        assert get_model_variant("glm-4-9b") is not None


# ---------------------------------------------------------------------------
# Workload loading reliability
# ---------------------------------------------------------------------------


class TestWorkloadLoading:
    """Verify all built-in workloads load without error."""

    def test_all_workloads_load(self) -> None:
        workloads = list_builtin_workloads()
        assert len(workloads) > 0
        for name in workloads:
            pack = load_workload(name)
            assert pack is not None
            assert len(pack.requests) > 0, f"Workload '{name}' has no requests"

    def test_production_disagg_coding_is_hydrated(self) -> None:
        """Production disagg workload must have real content, not template stubs."""
        pack = load_workload("production-disagg-coding")
        assert pack is not None
        assert "hydration:hydrated" in pack.tags
        assert "hydration:template" not in pack.tags
        # Verify all requests have substantive content
        for request in pack.requests:
            total_content = sum(
                len(msg.content) if isinstance(msg.content, str) else 0
                for msg in request.messages
            )
            assert total_content > 100, (
                f"Request '{request.name}' has insufficient content ({total_content} chars)"
            )

    def test_production_workload_has_sessions(self) -> None:
        """Production workload should have multiple sessions for disagg testing."""
        pack = load_workload("production-disagg-coding")
        sessions = {r.session_id for r in pack.requests if r.session_id}
        assert len(sessions) >= 2, "Need multiple sessions to test disaggregated routing"


# ---------------------------------------------------------------------------
# Experiment spec loading reliability
# ---------------------------------------------------------------------------


class TestExperimentSpecs:
    """Verify all built-in experiments load and are self-consistent."""

    def test_all_experiments_load(self) -> None:
        experiments = list_builtin_experiments()
        assert len(experiments) > 0
        for name in experiments:
            spec = load_experiment(name)
            assert spec is not None
            assert spec.engine in {"vllm", "sglang", "dynamo", "trtllm", "atom"}

    def test_all_experiments_reference_valid_workloads(self) -> None:
        """Every experiment must reference a workload that actually exists."""
        experiments = list_builtin_experiments()
        workloads = set(list_builtin_workloads())
        for name in experiments:
            spec = load_experiment(name)
            assert spec.workload in workloads, (
                f"Experiment '{name}' references workload '{spec.workload}' "
                f"which is not in the built-in workload list"
            )

    @pytest.mark.parametrize(
        "experiment_name",
        [
            "vllm-disagg-prefill-lmcache-kimi-k2",
            "dynamo-disagg-prefill-nixl-glm5",
            "vllm-disagg-prefill-lmcache-glm5",
        ],
    )
    def test_new_experiments_load(self, experiment_name: str) -> None:
        spec = load_experiment(experiment_name)
        assert spec is not None
        assert spec.model != ""
        assert spec.topology.mode == "prefill_decode_split"

    def test_kimi_k2_experiment_references_production_workload(self) -> None:
        spec = load_experiment("vllm-disagg-prefill-lmcache-kimi-k2")
        assert spec.workload == "production-disagg-coding"
        assert spec.model == "Kimi-K2"

    def test_glm5_dynamo_experiment(self) -> None:
        spec = load_experiment("dynamo-disagg-prefill-nixl-glm5")
        assert spec.workload == "production-disagg-coding"
        assert spec.model == "GLM-5"
        assert spec.engine == "dynamo"

    def test_glm5_lmcache_experiment(self) -> None:
        spec = load_experiment("vllm-disagg-prefill-lmcache-glm5")
        assert spec.workload == "production-disagg-coding"
        assert spec.model == "GLM-5"
        assert spec.engine == "vllm"


# ---------------------------------------------------------------------------
# Stack plan generation reliability
# ---------------------------------------------------------------------------


class TestStackPlanGeneration:
    """Verify stack plans generate successfully for target configurations."""

    @pytest.mark.parametrize(
        "experiment,model,gpu,num_gpus",
        [
            ("vllm-disagg-prefill-lmcache-kimi-k2", "Kimi-K2", "h100", 8),
            ("vllm-disagg-prefill-lmcache-glm5", "GLM-5", "h100", 4),
            ("dynamo-disagg-prefill-nixl-glm5", "GLM-5", "h100", 4),
            ("vllm-disagg-prefill-lmcache", "Kimi-K2", "h100", 8),
            ("vllm-disagg-prefill-lmcache", "GLM-5", "h100", 4),
            ("dynamo-disagg-prefill-nixl", "Kimi-K2.5", "h100", 8),
        ],
    )
    def test_stack_plan_generates(
        self, experiment: str, model: str, gpu: str, num_gpus: int
    ) -> None:
        """Stack plan generation must not raise for supported configurations."""
        plan = build_benchmark_stack_plan(
            experiment, gpu, num_gpus, model=model,
        )
        assert plan is not None
        assert plan.model == model
        assert plan.engine in {"vllm", "dynamo"}
        assert len(plan.components) >= 2, "Disaggregated plans need at least prefill + decode"
        assert plan.benchmark_command != ""

    @pytest.mark.parametrize(
        "experiment,model",
        [
            ("vllm-disagg-prefill-lmcache-kimi-k2", "Kimi-K2"),
            ("vllm-disagg-prefill-lmcache-glm5", "GLM-5"),
        ],
    )
    def test_lmcache_config_generated(self, experiment: str, model: str) -> None:
        """LMCache disagg plans must generate prefiller and decoder config files."""
        plan = build_benchmark_stack_plan(experiment, "h100", 4, model=model)
        config_paths = [f.path for f in plan.generated_files]
        assert any("prefiller" in p for p in config_paths), "Missing LMCache prefiller config"
        assert any("decoder" in p for p in config_paths), "Missing LMCache decoder config"

    @pytest.mark.parametrize(
        "experiment,model",
        [
            ("dynamo-disagg-prefill-nixl-glm5", "GLM-5"),
        ],
    )
    def test_dynamo_config_generated(self, experiment: str, model: str) -> None:
        """Dynamo disagg plans must generate a Dynamo deployment config."""
        plan = build_benchmark_stack_plan(experiment, "h100", 4, model=model)
        config_paths = [f.path for f in plan.generated_files]
        assert any("dynamo-config" in p for p in config_paths), "Missing Dynamo config"

    def test_stack_plan_has_readiness_probes(self) -> None:
        """Every engine component should have a readiness probe."""
        plan = build_benchmark_stack_plan(
            "vllm-disagg-prefill-lmcache-kimi-k2", "h100", 8, model="Kimi-K2",
        )
        engine_components = [c for c in plan.components if c.kind == "engine"]
        for component in engine_components:
            assert component.readiness is not None, (
                f"Component '{component.name}' missing readiness probe"
            )

    def test_stack_plan_component_ordering(self) -> None:
        """Prefill must come before decode in component list (launch order matters)."""
        plan = build_benchmark_stack_plan(
            "vllm-disagg-prefill-lmcache-kimi-k2", "h100", 8, model="Kimi-K2",
        )
        names = [c.name for c in plan.components]
        prefill_idx = next(i for i, n in enumerate(names) if "prefill" in n)
        decode_idx = next(i for i, n in enumerate(names) if "decode" in n)
        assert prefill_idx < decode_idx, "Prefill must launch before decode"


# ---------------------------------------------------------------------------
# Support assessment reliability
# ---------------------------------------------------------------------------


class TestSupportAssessment:
    """Verify support assessment gives correct results for target configs."""

    @pytest.mark.parametrize(
        "model_name,gpu_name",
        [
            ("Kimi-K2", "h100"),
            ("GLM-5", "h100"),
            ("GLM-4-9B", "h100"),
            ("Kimi-K2.5", "h100"),
        ],
    )
    def test_known_model_not_unsupported(self, model_name: str, gpu_name: str) -> None:
        """Known models on H100 should not be flagged as unsupported."""
        support = assess_benchmark_support(
            model_name=model_name,
            gpu_name=gpu_name,
            engine_name="vllm",
        )
        assert support.status != "unsupported", (
            f"{model_name} on {gpu_name} should not be unsupported. "
            f"Issues: {[i.message for i in support.issues]}"
        )

    def test_unknown_model_is_warning_not_error(self) -> None:
        """Unknown models should produce a warning, not block benchmarks."""
        support = assess_benchmark_support(
            model_name="SomeNewModel-999B",
            gpu_name="h100",
            engine_name="vllm",
        )
        error_codes = [i.code for i in support.issues if i.severity == "error"]
        assert "unknown_model" not in error_codes, "unknown_model should be warning, not error"


# ---------------------------------------------------------------------------
# BenchmarkConnectionError export
# ---------------------------------------------------------------------------


class TestConnectionError:
    def test_connection_error_is_runtime_error(self) -> None:
        """BenchmarkConnectionError must be catchable as RuntimeError."""
        exc = BenchmarkConnectionError("test")
        assert isinstance(exc, RuntimeError)
