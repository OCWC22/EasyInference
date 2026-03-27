"""Unit tests for BenchmarkCacheMetadata and compression validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from inferscope.benchmarks.experiments import (
    BenchmarkCacheCompressionMetadata,
    BenchmarkCacheMetadata,
)


def test_cache_metadata_defaults_backward_compatible() -> None:
    """Old-style cache metadata (no remote_backend/compression) still loads."""
    cache = BenchmarkCacheMetadata(
        strategy="lmcache",
        tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
        connector="LMCache",
    )
    assert cache.remote_backend == "unknown"
    assert cache.compression.enabled is False
    assert cache.compression.algorithm == "none"


def test_cache_metadata_simm_legacy_read_compat() -> None:
    """Legacy artifacts with remote_backend='simm' still deserialize (no validator rejects them)."""
    cache = BenchmarkCacheMetadata(
        strategy="lmcache",
        tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
        connector="LMCache",
        remote_backend="simm",
    )
    assert cache.remote_backend == "simm"


def test_cache_metadata_simm_hicache_legacy_read_compat() -> None:
    """Legacy HiCache + SiMM artifacts also deserialize."""
    cache = BenchmarkCacheMetadata(
        strategy="hicache",
        tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
        connector="HiCache",
        remote_backend="simm",
    )
    assert cache.remote_backend == "simm"


def test_compression_requires_algorithm() -> None:
    with pytest.raises(ValidationError, match="no algorithm"):
        BenchmarkCacheMetadata(
            strategy="lmcache",
            tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
            remote_backend="none",
            compression=BenchmarkCacheCompressionMetadata(
                enabled=True,
                algorithm="none",
            ),
        )


def test_compression_applies_to_must_be_in_tiers() -> None:
    with pytest.raises(ValidationError, match="not in cache tiers"):
        BenchmarkCacheMetadata(
            strategy="lmcache",
            tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
            remote_backend="none",
            compression=BenchmarkCacheCompressionMetadata(
                enabled=True,
                algorithm="lz4",
                applies_to=["local_ssd"],  # not in tiers
            ),
        )


def test_compression_lz4_legacy_valid() -> None:
    """Legacy artifact with compression on SiMM still deserializes."""
    cache = BenchmarkCacheMetadata(
        strategy="lmcache",
        tiers=["gpu_hbm", "grace_coherent", "remote_cache"],
        connector="LMCache",
        remote_backend="simm",
        compression=BenchmarkCacheCompressionMetadata(
            enabled=True,
            algorithm="lz4",
            applies_to=["remote_cache"],
        ),
    )
    assert cache.compression.enabled is True
    assert cache.compression.algorithm == "lz4"


def test_compression_disabled_allows_any_backend() -> None:
    """Disabled compression should not trigger validation errors."""
    cache = BenchmarkCacheMetadata(
        strategy="lmcache",
        tiers=["gpu_hbm", "cpu_dram", "remote_cache"],
        remote_backend="unknown",
    )
    assert cache.compression.enabled is False
