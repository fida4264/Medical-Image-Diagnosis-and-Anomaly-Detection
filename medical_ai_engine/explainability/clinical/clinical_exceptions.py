"""
clinical_exceptions.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Typed exception hierarchy for the clinical explanation subsystem.
"""

from __future__ import annotations


class ClinicalExplanationError(Exception):
    """Base exception for the Clinical Explanation Engine."""

    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(message)
        self.details: dict = details or {}

    def __str__(self) -> str:  # pragma: no cover
        base = super().__str__()
        if self.details:
            return f"{base} | details={self.details}"
        return base


# ── Sub-exceptions ──────────────────────────────────────────────────────────

class FindingClassificationError(ClinicalExplanationError):
    """Raised when anatomical finding classification fails."""


class SeverityAssessmentError(ClinicalExplanationError):
    """Raised when severity level assessment cannot be determined."""


class RegionDescriptionError(ClinicalExplanationError):
    """Raised when spatial region description generation fails."""


class TextGenerationError(ClinicalExplanationError):
    """Raised when natural-language clinical text generation fails."""


class RecommendationEngineError(ClinicalExplanationError):
    """Raised when clinical recommendation generation fails."""


class ReportBuildError(ClinicalExplanationError):
    """Raised when the clinical report cannot be assembled."""


class ClinicalContextError(ClinicalExplanationError):
    """Raised when the clinical context is invalid or missing required fields."""


class ConfigurationError(ClinicalExplanationError):
    """Raised when the clinical configuration is malformed."""


__all__ = [
    "ClinicalExplanationError",
    "FindingClassificationError",
    "SeverityAssessmentError",
    "RegionDescriptionError",
    "TextGenerationError",
    "RecommendationEngineError",
    "ReportBuildError",
    "ClinicalContextError",
    "ConfigurationError",
]
