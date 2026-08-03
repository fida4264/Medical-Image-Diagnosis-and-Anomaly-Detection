"""
__init__.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Benchmark package public API.
"""

from __future__ import annotations

from medical_ai_engine.benchmark.benchmark_configuration import BenchmarkConfiguration
from medical_ai_engine.benchmark.benchmark_exceptions import (
    BenchmarkConfigurationError,
    BenchmarkException,
    BenchmarkExecutionError,
    BenchmarkReportError,
    ComparisonEngineError,
    LeaderboardError,
    MetricComputationError,
)
from medical_ai_engine.benchmark.benchmark_report import BenchmarkReport
from medical_ai_engine.benchmark.benchmark_runner import BenchmarkRunner
from medical_ai_engine.benchmark.comparison_engine import (
    ComparisonEngine,
    ComparisonSummary,
    ModelComparisonEntry,
)
from medical_ai_engine.benchmark.leaderboard import (
    Leaderboard,
    LeaderboardBuilder,
    LeaderboardRow,
)
from medical_ai_engine.benchmark.metrics import (
    ClassificationMetrics,
    ModalityBenchmarkResult,
    PerformanceMetrics,
    calculate_classification_metrics,
)
from medical_ai_engine.benchmark.performance_profiler import PerformanceProfiler

__all__ = [
    # Configuration
    "BenchmarkConfiguration",
    # Runner & Profiler
    "BenchmarkRunner",
    "PerformanceProfiler",
    # Metrics
    "ClassificationMetrics",
    "PerformanceMetrics",
    "ModalityBenchmarkResult",
    "calculate_classification_metrics",
    # Comparison
    "ComparisonEngine",
    "ComparisonSummary",
    "ModelComparisonEntry",
    # Leaderboard
    "Leaderboard",
    "LeaderboardBuilder",
    "LeaderboardRow",
    # Report
    "BenchmarkReport",
    # Exceptions
    "BenchmarkException",
    "BenchmarkExecutionError",
    "MetricComputationError",
    "ComparisonEngineError",
    "LeaderboardError",
    "BenchmarkReportError",
    "BenchmarkConfigurationError",
]
