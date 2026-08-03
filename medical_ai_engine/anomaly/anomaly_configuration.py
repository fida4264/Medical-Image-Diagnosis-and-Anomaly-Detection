"""
anomaly_configuration.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Immutable configuration dataclass for the Anomaly Detection Framework.

This module defines ARCHITECTURE-ONLY configuration.
No algorithm parameters are defined here (belongs to Module 16.2+).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .anomaly_exceptions import AnomalyConfigurationError


@dataclass
class AnomalyConfiguration:
    """
    Production-grade configuration for the Anomaly Detection Framework.

    Attributes
    ----------
    enabled : bool
        Master on/off switch. If False, all anomaly detection is bypassed.
    fail_silently : bool
        If True, anomaly detection failures never break primary prediction.
    async_execution : bool
        Run anomaly detection in a non-blocking background thread.
    default_strategy : str
        Strategy key to use when no override is specified.
        Must match a registered IDetectionStrategy key.
    confidence_threshold_low : float
        Below this confidence score, prediction is flagged LOW_CONFIDENCE.
    confidence_threshold_critical : float
        Below this confidence score, prediction is flagged CRITICAL.
    quality_score_threshold : float
        Below this score, image is flagged as POOR_QUALITY.
    ood_distance_threshold : float
        Mahalanobis / feature distance threshold for OOD decisions.
    reconstruction_error_threshold : float
        Autoencoder MSE threshold for UNKNOWN_DISEASE detection.
    supported_modalities : List[str]
        Modalities that trigger anomaly analysis; empty = all.
    enable_confidence_analysis : bool
        Enable the Confidence Analyzer sub-component.
    enable_ood_detection : bool
        Enable the OOD Detector sub-component.
    enable_unknown_disease_detection : bool
        Enable the Unknown Disease Detector sub-component.
    enable_risk_assessment : bool
        Enable Risk Assessment after anomaly detection.
    max_processing_time_ms : float
        Timeout for anomaly detection pipeline execution.
    """

    enabled: bool = True
    fail_silently: bool = True
    async_execution: bool = False

    # Strategy selection
    default_strategy: str = "confidence_threshold"

    # Confidence thresholds
    confidence_threshold_low: float = 0.70
    confidence_threshold_critical: float = 0.40

    # Quality threshold
    quality_score_threshold: float = 0.30

    # OOD & reconstruction thresholds
    ood_distance_threshold: float = 3.0
    reconstruction_error_threshold: float = 0.10

    # Sub-component toggles
    enable_confidence_analysis: bool = True
    enable_ood_detection: bool = True
    enable_unknown_disease_detection: bool = True
    enable_risk_assessment: bool = True

    # Timeout
    max_processing_time_ms: float = 50.0

    # Supported modalities (empty = all)
    supported_modalities: List[str] = field(
        default_factory=lambda: [
            "brain_mri", "chest_xray", "brain_ct",
            "histopathology", "retina", "skin", "ultrasound",
        ]
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if not 0.0 <= self.confidence_threshold_low <= 1.0:
            raise AnomalyConfigurationError(
                f"confidence_threshold_low must be in [0, 1]; got {self.confidence_threshold_low}"
            )
        if not 0.0 <= self.confidence_threshold_critical <= 1.0:
            raise AnomalyConfigurationError(
                f"confidence_threshold_critical must be in [0, 1]; got {self.confidence_threshold_critical}"
            )
        if self.confidence_threshold_critical > self.confidence_threshold_low:
            raise AnomalyConfigurationError(
                "confidence_threshold_critical must be <= confidence_threshold_low"
            )
        if self.max_processing_time_ms < 0:
            raise AnomalyConfigurationError(
                f"max_processing_time_ms must be >= 0; got {self.max_processing_time_ms}"
            )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "fail_silently": self.fail_silently,
            "async_execution": self.async_execution,
            "default_strategy": self.default_strategy,
            "confidence_threshold_low": self.confidence_threshold_low,
            "confidence_threshold_critical": self.confidence_threshold_critical,
            "quality_score_threshold": self.quality_score_threshold,
            "ood_distance_threshold": self.ood_distance_threshold,
            "reconstruction_error_threshold": self.reconstruction_error_threshold,
            "enable_confidence_analysis": self.enable_confidence_analysis,
            "enable_ood_detection": self.enable_ood_detection,
            "enable_unknown_disease_detection": self.enable_unknown_disease_detection,
            "enable_risk_assessment": self.enable_risk_assessment,
            "max_processing_time_ms": self.max_processing_time_ms,
        }

    @classmethod
    def default(cls) -> "AnomalyConfiguration":
        """Create a default production configuration."""
        return cls()

    @classmethod
    def fast_test(cls) -> "AnomalyConfiguration":
        """Lightweight configuration optimized for unit testing."""
        return cls(
            enabled=True,
            fail_silently=True,
            async_execution=False,
            confidence_threshold_low=0.70,
            confidence_threshold_critical=0.40,
            enable_confidence_analysis=True,
            enable_ood_detection=True,
            enable_unknown_disease_detection=True,
            enable_risk_assessment=True,
            max_processing_time_ms=500.0,
        )

    @classmethod
    def disabled(cls) -> "AnomalyConfiguration":
        """Create a disabled configuration (bypasses all anomaly detection)."""
        return cls(enabled=False)


__all__ = ["AnomalyConfiguration"]
