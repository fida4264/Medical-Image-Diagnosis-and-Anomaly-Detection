"""
Medical AI Engine — Verification Configuration

Module 14.2.7: Production Verification & Stress Testing

Centralized configuration for all production verification parameters.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class VerificationConfiguration:
    """
    All tunable parameters for Module 14.2.7 production verification.
    """

    # ── Supported Modalities ──────────────────────────────────────────────────
    supported_modalities: List[str] = field(default_factory=lambda: [
        "brain_mri",
        "chest_xray",
        "brain_ct",
        "histopathology",
        "retina",
        "skin",
        "ultrasound",
    ])

    # ── Thread Safety ─────────────────────────────────────────────────────────
    thread_concurrency_levels: List[int] = field(default_factory=lambda: [20, 50, 100])
    thread_test_timeout_seconds: float = 60.0
    thread_barrier_timeout_seconds: float = 10.0

    # ── Memory Profiling ──────────────────────────────────────────────────────
    memory_test_duration_seconds: float = 30.0
    memory_test_cycles: int = 100
    memory_leak_threshold_objects: int = 500
    memory_leak_threshold_mb: float = 50.0

    # ── Performance Benchmark ─────────────────────────────────────────────────
    benchmark_iterations: int = 50
    benchmark_warmup_iterations: int = 5
    latency_threshold_avg_ms: float = 500.0
    latency_threshold_p95_ms: float = 1000.0
    min_throughput_rps: float = 2.0

    # ── Stress Testing ────────────────────────────────────────────────────────
    stress_burst_count: int = 50
    stress_sustained_count: int = 100
    stress_sustained_duration_seconds: float = 20.0
    stress_ramp_levels: List[int] = field(default_factory=lambda: [10, 20, 50, 100])
    stress_test_timeout_seconds: float = 120.0

    # ── Architecture Audit ────────────────────────────────────────────────────
    check_circular_imports: bool = True
    check_interface_adherence: bool = True
    check_design_patterns: bool = True
    check_module_isolation: bool = True
    max_allowed_critical_violations: int = 0

    # ── Compatibility ─────────────────────────────────────────────────────────
    check_prediction_result_fields: bool = True
    check_engine_context_fields: bool = True
    check_enum_values: bool = True
    check_json_roundtrip: bool = True
    check_adapter_output_keys: bool = True

    # ── Health Monitoring ─────────────────────────────────────────────────────
    health_check_interval_seconds: float = 1.0
    health_score_pass_threshold: float = 60.0
    failure_rate_warning_threshold: float = 0.10

    # ── Quality Scoring ───────────────────────────────────────────────────────
    quality_pass_threshold: float = 70.0
    overall_pass_threshold: float = 75.0

    # ── Reporting ─────────────────────────────────────────────────────────────
    report_output_path: str = "production_readiness_report.md"
    verbose: bool = True
    use_mock_fallbacks: bool = True

    @classmethod
    def default(cls) -> "VerificationConfiguration":
        """Return the default production verification configuration."""
        return cls()

    @classmethod
    def fast(cls) -> "VerificationConfiguration":
        """Return a fast configuration suitable for CI/CD pipelines."""
        return cls(
            thread_concurrency_levels=[20, 50],
            memory_test_cycles=30,
            memory_test_duration_seconds=10.0,
            benchmark_iterations=20,
            benchmark_warmup_iterations=2,
            stress_burst_count=20,
            stress_sustained_count=40,
            stress_sustained_duration_seconds=10.0,
            stress_ramp_levels=[10, 20, 50],
            stress_test_timeout_seconds=60.0,
            thread_test_timeout_seconds=30.0,
        )

    def to_dict(self) -> dict:
        """Serialize configuration to dict."""
        return {k: v for k, v in self.__dict__.items()}
