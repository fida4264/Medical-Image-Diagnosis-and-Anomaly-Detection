"""
verification_exceptions.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Typed exception hierarchy for the XAI verification subsystem.
"""

from __future__ import annotations


class ExplainabilityVerificationError(Exception):
    """Base exception for Explainability Verification & Benchmarking failures."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details: dict = details or {}


class ArchitectureValidationError(ExplainabilityVerificationError):
    """Raised when architecture compliance verification fails."""


class GradCAMVerificationError(ExplainabilityVerificationError):
    """Raised when Grad-CAM engine validation fails."""


class VisualizationVerificationError(ExplainabilityVerificationError):
    """Raised when visualization rendering validation fails."""


class ClinicalExplanationVerificationError(ExplainabilityVerificationError):
    """Raised when clinical explanation validation fails."""


class BenchmarkExecutionError(ExplainabilityVerificationError):
    """Raised when performance benchmark or profiling fails."""


class StressTestError(ExplainabilityVerificationError):
    """Raised when concurrent stress testing fails or deadlocks."""


class QualityEvaluationError(ExplainabilityVerificationError):
    """Raised when quality or faithfulness evaluation fails."""


class RegressionTestError(ExplainabilityVerificationError):
    """Raised when prior module regression testing fails."""


class ProductionVerificationError(ExplainabilityVerificationError):
    """Raised when overall production readiness validation fails."""


__all__ = [
    "ExplainabilityVerificationError",
    "ArchitectureValidationError",
    "GradCAMVerificationError",
    "VisualizationVerificationError",
    "ClinicalExplanationVerificationError",
    "BenchmarkExecutionError",
    "StressTestError",
    "QualityEvaluationError",
    "RegressionTestError",
    "ProductionVerificationError",
]
