"""
clinical_configuration.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Immutable configuration dataclass for the clinical text subsystem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .clinical_exceptions import ConfigurationError

# ── Severity thresholds ──────────────────────────────────────────────────────
DEFAULT_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "critical":  0.85,
    "high":      0.65,
    "moderate":  0.40,
    "low":       0.20,
    "minimal":   0.0,
}

# ── Region size bins (fraction of image area) ────────────────────────────────
DEFAULT_REGION_BINS: Dict[str, float] = {
    "extensive":  0.40,
    "large":      0.25,
    "moderate":   0.12,
    "small":      0.05,
    "focal":      0.0,
}

# ── Supported modalities ─────────────────────────────────────────────────────
SUPPORTED_MODALITIES: List[str] = [
    "brain_mri", "chest_xray", "brain_ct",
    "histopathology", "retina", "skin", "ultrasound",
]


@dataclass
class ClinicalConfiguration:
    """
    Immutable configuration for the Clinical Explanation Engine.

    Attributes
    ----------
    language:
        Output language for generated text (ISO 639-1; default 'en').
    include_recommendations:
        Whether to append clinical action recommendations.
    include_confidence_statement:
        Prepend AI confidence disclaimer to each report.
    include_disclaimer:
        Append regulatory / not-for-clinical-use disclaimer.
    max_findings_per_report:
        Cap on number of distinct findings included in one report.
    severity_thresholds:
        Mapping from severity label → minimum activation score.
    region_bins:
        Mapping from region size label → minimum area fraction.
    verbosity:
        Output verbosity: 'brief' | 'standard' | 'detailed'.
    """

    language: str = "en"
    include_recommendations: bool = True
    include_confidence_statement: bool = True
    include_disclaimer: bool = True
    max_findings_per_report: int = 5
    severity_thresholds: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SEVERITY_THRESHOLDS)
    )
    region_bins: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_REGION_BINS)
    )
    verbosity: str = "standard"

    def __post_init__(self) -> None:
        self._validate()

    # ── Validation ───────────────────────────────────────────────────────────

    def _validate(self) -> None:
        if self.language not in {"en"}:
            raise ConfigurationError(
                f"Unsupported language '{self.language}'. Only 'en' is currently supported."
            )
        if self.verbosity not in {"brief", "standard", "detailed"}:
            raise ConfigurationError(
                f"Invalid verbosity '{self.verbosity}'. Must be one of: brief, standard, detailed."
            )
        if not 1 <= self.max_findings_per_report <= 20:
            raise ConfigurationError(
                f"max_findings_per_report must be between 1 and 20, got {self.max_findings_per_report}."
            )
        if not self.severity_thresholds:
            raise ConfigurationError("severity_thresholds must not be empty.")
        if not self.region_bins:
            raise ConfigurationError("region_bins must not be empty.")

    # ── Helpers ──────────────────────────────────────────────────────────────

    def classify_severity(self, score: float) -> str:
        """Return the severity label for a given normalized activation score."""
        for label, threshold in sorted(
            self.severity_thresholds.items(), key=lambda kv: kv[1], reverse=True
        ):
            if score >= threshold:
                return label
        return "minimal"

    def classify_region_size(self, area_fraction: float) -> str:
        """Return the region size label for a given area fraction."""
        for label, threshold in sorted(
            self.region_bins.items(), key=lambda kv: kv[1], reverse=True
        ):
            if area_fraction >= threshold:
                return label
        return "focal"

    def to_dict(self) -> dict:
        return {
            "language": self.language,
            "include_recommendations": self.include_recommendations,
            "include_confidence_statement": self.include_confidence_statement,
            "include_disclaimer": self.include_disclaimer,
            "max_findings_per_report": self.max_findings_per_report,
            "verbosity": self.verbosity,
        }


__all__ = [
    "ClinicalConfiguration",
    "DEFAULT_SEVERITY_THRESHOLDS",
    "DEFAULT_REGION_BINS",
    "SUPPORTED_MODALITIES",
]
