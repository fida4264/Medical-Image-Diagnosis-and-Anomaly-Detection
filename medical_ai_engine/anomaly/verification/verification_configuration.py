"""
verification_configuration.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Configuration dataclass for the verification subsystem.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from .verification_exceptions import VerificationConfigurationError


@dataclass
class VerificationConfiguration:
    """
    Configuration for the full Module 16.7 Verification & Benchmarking suite.

    Attributes
    ----------
    n_warmup_iterations : int
        Warmup runs before benchmark timing begins.
    n_benchmark_iterations : int
        Measurement iterations for benchmark statistics.
    stress_thread_counts : List[int]
        Thread counts for stress-test tiers (e.g. [20, 50, 100]).
    latency_budget_ms : float
        Maximum acceptable end-to-end anomaly latency per request.
    throughput_target_rps : float
        Minimum acceptable requests-per-second throughput.
    memory_budget_mb : float
        Maximum acceptable RSS memory growth during a benchmark run.
    min_pass_rate : float
        Minimum fraction of requests that must succeed (0.0–1.0).
    enable_memory_profiling : bool
        Whether to collect RSS memory snapshots.
    enable_cpu_profiling : bool
        Whether to collect CPU usage snapshots.
    supported_modalities : List[str]
        Modalities exercised in the verification suite.
    """

    n_warmup_iterations: int = 5
    n_benchmark_iterations: int = 100
    stress_thread_counts: List[int] = field(default_factory=lambda: [20, 50, 100])
    latency_budget_ms: float = 50.0
    throughput_target_rps: float = 20.0
    memory_budget_mb: float = 200.0
    min_pass_rate: float = 0.99
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri",
            "chest_xray",
            "histopathology",
            "retina",
            "skin",
            "ultrasound",
            "brain_ct",
        ]
    )

    def __post_init__(self) -> None:
        if self.n_benchmark_iterations < 1:
            raise VerificationConfigurationError(
                f"n_benchmark_iterations must be >= 1; got {self.n_benchmark_iterations}"
            )
        if self.latency_budget_ms <= 0:
            raise VerificationConfigurationError(
                f"latency_budget_ms must be > 0; got {self.latency_budget_ms}"
            )
        if not (0.0 < self.min_pass_rate <= 1.0):
            raise VerificationConfigurationError(
                f"min_pass_rate must be in (0, 1]; got {self.min_pass_rate}"
            )

    @classmethod
    def default(cls) -> "VerificationConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "VerificationConfiguration":
        """Reduced iteration counts for unit tests."""
        return cls(
            n_warmup_iterations=2,
            n_benchmark_iterations=10,
            stress_thread_counts=[5, 10],
            latency_budget_ms=500.0,
            throughput_target_rps=5.0,
            memory_budget_mb=500.0,
        )


__all__ = ["VerificationConfiguration"]
