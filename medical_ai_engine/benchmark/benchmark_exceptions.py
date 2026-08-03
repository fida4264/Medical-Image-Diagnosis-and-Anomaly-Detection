"""
benchmark_exceptions.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Typed exception hierarchy for the benchmark subsystem.
"""

from __future__ import annotations


class BenchmarkException(Exception):
    """Base exception for all benchmarking subsystem errors."""


class BenchmarkExecutionError(BenchmarkException):
    """Raised when benchmark runner execution encounters a failure."""


class MetricComputationError(BenchmarkException):
    """Raised when calculation of classification or performance metrics fails."""


class ComparisonEngineError(BenchmarkException):
    """Raised when model or modality comparison analysis fails."""


class LeaderboardError(BenchmarkException):
    """Raised when constructing or formatting leaderboard fails."""


class BenchmarkReportError(BenchmarkException):
    """Raised when generating benchmark markdown or JSON reports fails."""


class BenchmarkConfigurationError(BenchmarkException):
    """Raised when benchmark configuration options are invalid."""


__all__ = [
    "BenchmarkException",
    "BenchmarkExecutionError",
    "MetricComputationError",
    "ComparisonEngineError",
    "LeaderboardError",
    "BenchmarkReportError",
    "BenchmarkConfigurationError",
]
