"""
verification_exceptions.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Typed exception hierarchy for verification subsystem.
"""
from __future__ import annotations


class VerificationError(Exception):
    """Base exception for all verification errors."""


class BenchmarkError(VerificationError):
    """Raised when benchmarking fails to collect valid metrics."""


class StressTestError(VerificationError):
    """Raised when a stress test detects a failure condition."""


class MemoryProfilerError(VerificationError):
    """Raised when memory profiling encounters an error."""


class QualityValidationError(VerificationError):
    """Raised when quality validation thresholds are not met."""


class IntegrationValidationError(VerificationError):
    """Raised when integration validation fails."""


class ProductionValidationError(VerificationError):
    """Raised when production readiness criteria are not met."""


class VerificationConfigurationError(VerificationError):
    """Raised when verification configuration is invalid."""


class ReportGenerationError(VerificationError):
    """Raised when report generation fails."""


__all__ = [
    "VerificationError",
    "BenchmarkError",
    "StressTestError",
    "MemoryProfilerError",
    "QualityValidationError",
    "IntegrationValidationError",
    "ProductionValidationError",
    "VerificationConfigurationError",
    "ReportGenerationError",
]
