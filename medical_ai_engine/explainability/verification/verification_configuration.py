"""
verification_configuration.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Configuration dataclass for the verification suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class VerificationConfiguration:
    """
    Configuration settings for Explainability Framework verification & benchmarking.
    """

    stress_concurrency_levels: List[int] = field(
        default_factory=lambda: [20, 50, 100]
    )
    stress_iterations_per_level: int = 10
    max_latency_p95_ms: float = 100.0
    max_memory_mb: float = 1024.0
    min_quality_score: float = 70.0
    min_sharpness_score: float = 0.05
    min_coverage_ratio: float = 0.05
    max_coverage_ratio: float = 0.85
    enable_memory_profiling: bool = True
    enable_quality_evaluation: bool = True
    enable_faithfulness_evaluation: bool = True
    enable_stress_testing: bool = True
    export_report_markdown: bool = True
    report_output_dir: str = "artifacts/reports"
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri", "chest_xray", "brain_ct",
            "histopathology", "retina", "skin", "ultrasound",
        ]
    )

    @classmethod
    def default(cls) -> "VerificationConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "VerificationConfiguration":
        """Fast configuration for unit tests."""
        return cls(
            stress_concurrency_levels=[5, 10],
            stress_iterations_per_level=2,
            enable_memory_profiling=True,
            enable_quality_evaluation=True,
            enable_faithfulness_evaluation=True,
            enable_stress_testing=True,
            export_report_markdown=False,
        )


__all__ = ["VerificationConfiguration"]
