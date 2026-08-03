"""
benchmark_configuration.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Configuration dataclass for global platform benchmarking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from medical_ai_engine.benchmark.benchmark_exceptions import BenchmarkConfigurationError


@dataclass
class BenchmarkConfiguration:
    """
    Configuration options for global model evaluation and benchmarking.

    Attributes
    ----------
    supported_modalities : List[str]
        All 7 platform modalities to evaluate.
    n_warmup_runs : int
        Number of un-timed warmup inferences before profiling.
    n_eval_runs : int
        Number of evaluation iterations per model for latency/throughput profiling.
    batch_sizes : List[int]
        Batch sizes to evaluate for throughput scaling analysis.
    output_directory : str
        Directory to save markdown and JSON benchmark reports.
    device : str
        Target hardware device ('cpu', 'cuda', etc.).
    compute_roc_auc : bool
        Whether to compute multi-class / binary ROC-AUC.
    enable_memory_profiling : bool
        Whether to track RSS memory baseline, peak, and growth.
    enable_cpu_profiling : bool
        Whether to collect CPU usage statistics.
    latency_budget_ms : float
        Maximum target inference latency budget in milliseconds.
    throughput_target_rps : float
        Minimum target requests-per-second throughput.
    """

    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri",
            "chest_xray",
            "brain_ct",
            "histopathology",
            "retina",
            "skin",
            "ultrasound",
        ]
    )
    n_warmup_runs: int = 5
    n_eval_runs: int = 50
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32])
    output_directory: str = "docs"
    device: str = "cpu"
    compute_roc_auc: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    latency_budget_ms: float = 20.0
    throughput_target_rps: float = 50.0

    def __post_init__(self) -> None:
        if self.n_warmup_runs < 0:
            raise BenchmarkConfigurationError("n_warmup_runs must be >= 0")
        if self.n_eval_runs < 1:
            raise BenchmarkConfigurationError("n_eval_runs must be >= 1")
        if not self.supported_modalities:
            raise BenchmarkConfigurationError("supported_modalities cannot be empty")
        if self.latency_budget_ms <= 0:
            raise BenchmarkConfigurationError("latency_budget_ms must be > 0")

    @classmethod
    def default(cls) -> BenchmarkConfiguration:
        """Production default configuration."""
        return cls()

    @classmethod
    def fast_test(cls) -> BenchmarkConfiguration:
        """Fast configuration for unit testing."""
        return cls(
            n_warmup_runs=2,
            n_eval_runs=10,
            batch_sizes=[1, 4],
            latency_budget_ms=500.0,
            throughput_target_rps=5.0,
        )


__all__ = ["BenchmarkConfiguration"]
