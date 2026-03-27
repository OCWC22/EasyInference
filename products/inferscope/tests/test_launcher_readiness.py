"""Tests for launcher readiness probes and component ordering."""

from __future__ import annotations

import pytest

from inferscope.benchmarks.launchers import (
    BenchmarkStackPlan,
    LaunchComponent,
    LaunchReadinessProbe,
    _render_start_script,
)


# ---------------------------------------------------------------------------
# LaunchReadinessProbe model
# ---------------------------------------------------------------------------


def test_readiness_probe_defaults() -> None:
    probe = LaunchReadinessProbe(target="http://localhost:8000/health")
    assert probe.kind == "http"
    assert probe.timeout_seconds == 120
    assert probe.interval_seconds == 2.0


def test_readiness_probe_tcp() -> None:
    probe = LaunchReadinessProbe(kind="tcp", target="localhost:9000", timeout_seconds=30)
    assert probe.kind == "tcp"
    assert probe.timeout_seconds == 30


# ---------------------------------------------------------------------------
# LaunchComponent with readiness, depends_on, required_binaries
# ---------------------------------------------------------------------------


def test_component_readiness_and_depends_on() -> None:
    component = LaunchComponent(
        name="vllm-decode",
        role="decode",
        kind="engine",
        engine="vllm",
        command="vllm serve --port 7200",
        readiness=LaunchReadinessProbe(
            kind="http",
            target="http://localhost:7200/health",
            timeout_seconds=180,
        ),
        depends_on=["vllm-prefill"],
        required_binaries=["vllm"],
    )
    assert component.readiness is not None
    assert component.readiness.target == "http://localhost:7200/health"
    assert component.depends_on == ["vllm-prefill"]
    assert component.required_binaries == ["vllm"]


def test_component_defaults_no_readiness() -> None:
    component = LaunchComponent(
        name="simple",
        role="primary",
        kind="engine",
        command="echo hello",
    )
    assert component.readiness is None
    assert component.depends_on == []
    assert component.required_binaries == []


# ---------------------------------------------------------------------------
# _render_start_script with readiness probes
# ---------------------------------------------------------------------------


def test_render_start_script_without_probes() -> None:
    script = _render_start_script([("worker", "scripts/worker.sh")])
    assert "start_component worker" in script
    # wait_ready function is defined but should not be called for this component
    lines = [l.strip() for l in script.splitlines() if l.strip().startswith("wait_ready ")]
    assert lines == []


def test_render_start_script_with_http_probe() -> None:
    probes = {
        "worker": LaunchReadinessProbe(
            kind="http",
            target="http://localhost:8000/health",
            timeout_seconds=60,
            interval_seconds=1.0,
        ),
    }
    script = _render_start_script([("worker", "scripts/worker.sh")], readiness_probes=probes)
    assert "start_component worker" in script
    assert "wait_ready worker http" in script
    assert "http://localhost:8000/health" in script
    assert "60" in script


def test_render_start_script_with_tcp_probe() -> None:
    probes = {
        "proxy": LaunchReadinessProbe(
            kind="tcp",
            target="localhost:9000",
            timeout_seconds=30,
        ),
    }
    script = _render_start_script([("proxy", "scripts/proxy.sh")], readiness_probes=probes)
    assert "wait_ready proxy tcp" in script


def test_render_start_script_multiple_components_ordered() -> None:
    probes = {
        "prefill": LaunchReadinessProbe(target="http://localhost:7100/health", timeout_seconds=180),
        "decode": LaunchReadinessProbe(target="http://localhost:7200/health", timeout_seconds=180),
        "router": LaunchReadinessProbe(target="http://localhost:9100/metrics", timeout_seconds=60),
    }
    script = _render_start_script(
        [
            ("prefill", "scripts/prefill.sh"),
            ("decode", "scripts/decode.sh"),
            ("router", "scripts/router.sh"),
        ],
        readiness_probes=probes,
    )
    lines = script.splitlines()
    # Find the call lines (not the function definitions)
    start_lines = [i for i, l in enumerate(lines) if l.strip().startswith("start_component ")]
    wait_lines = [i for i, l in enumerate(lines) if l.strip().startswith("wait_ready ")]
    assert len(start_lines) == 3
    assert len(wait_lines) == 3
    # Each wait_ready comes after its start_component
    for s, w in zip(start_lines, wait_lines):
        assert w > s


def test_render_start_script_has_bash_functions() -> None:
    script = _render_start_script([("x", "scripts/x.sh")])
    assert "start_component()" in script
    assert "wait_ready()" in script
    assert "#!/usr/bin/env bash" in script
