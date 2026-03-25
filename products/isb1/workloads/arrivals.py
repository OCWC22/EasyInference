"""Arrival-time generators for ISB-1 benchmark workloads.

Provides stochastic inter-arrival time models that produce absolute
timestamps (in seconds) for when each request should be dispatched.
"""

from __future__ import annotations

import numpy as np


class PoissonArrival:
    """Generate request arrival times from a Poisson process.

    Inter-arrival times are drawn from an exponential distribution with
    rate parameter ``rate`` (requests per second).

    Parameters:
        rate: Mean number of requests per second (lambda).
        seed: Random seed for reproducibility.
    """

    def __init__(self, rate: float, seed: int = 42) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        self.rate = rate
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, num_requests: int) -> np.ndarray:
        """Return *num_requests* absolute arrival timestamps in seconds.

        The first arrival occurs at a time drawn from the same exponential
        distribution (i.e. the process starts at time 0, and the first
        event follows).

        Args:
            num_requests: Number of timestamps to produce.

        Returns:
            A 1-D ``numpy.ndarray`` of monotonically increasing floats
            representing arrival times in seconds.
        """
        if num_requests <= 0:
            return np.array([], dtype=np.float64)

        inter_arrivals = self.rng.exponential(
            scale=1.0 / self.rate, size=num_requests
        )
        return np.cumsum(inter_arrivals)


class GammaArrival:
    """Generate bursty request arrival times using a Gamma distribution.

    The Gamma distribution generalises the exponential: when the shape
    parameter *k* equals 1 it reduces to a Poisson process; values of
    *k* < 1 produce burstier traffic (higher variance relative to the
    mean), while *k* > 1 produces more regular traffic.

    Parameters:
        rate: Mean number of requests per second.
        shape: Gamma shape parameter *k*.  Smaller values create burstier
            arrival patterns (default 0.5).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self, rate: float, shape: float = 0.5, seed: int = 42
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if shape <= 0:
            raise ValueError(f"shape must be positive, got {shape}")
        self.rate = rate
        self.shape = shape
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def generate(self, num_requests: int) -> np.ndarray:
        """Return *num_requests* absolute arrival timestamps in seconds.

        Inter-arrival times are drawn from a Gamma distribution whose mean
        equals ``1 / rate`` so the long-run average throughput matches the
        requested rate.

        Args:
            num_requests: Number of timestamps to produce.

        Returns:
            A 1-D ``numpy.ndarray`` of monotonically increasing floats.
        """
        if num_requests <= 0:
            return np.array([], dtype=np.float64)

        # mean of Gamma(shape, scale) = shape * scale
        # We want mean = 1/rate  =>  scale = 1 / (rate * shape)
        scale = 1.0 / (self.rate * self.shape)
        inter_arrivals = self.rng.gamma(
            shape=self.shape, scale=scale, size=num_requests
        )
        return np.cumsum(inter_arrivals)
