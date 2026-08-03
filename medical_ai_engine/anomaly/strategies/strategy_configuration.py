"""
strategy_configuration.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Configuration options for confidence & Mahalanobis strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModalityThresholds:
    """Per-modality confidence thresholds."""
    high_confidence: float = 0.85
    medium_confidence: float = 0.70
    low_confidence: float = 0.50
    reject_threshold: float = 0.30
    mahalanobis_ood_threshold: float = 3.50


@dataclass
class StrategyConfiguration:
    """
    Configuration options for confidence & OOD strategies.

    Attributes
    ----------
    default_modality_thresholds : ModalityThresholds
        Global default thresholds when no per-modality override exists.
    modality_thresholds : Dict[str, ModalityThresholds]
        Per-modality threshold overrides.
    covariance_regularization : float
        Small positive constant added to covariance diagonal for numerical stability (ledoit-wolf / Tikhonov).
    """

    default_thresholds: ModalityThresholds = field(default_factory=ModalityThresholds)
    modality_thresholds: Dict[str, ModalityThresholds] = field(
        default_factory=lambda: {
            "brain_mri": ModalityThresholds(high_confidence=0.88, mahalanobis_ood_threshold=3.2),
            "chest_xray": ModalityThresholds(high_confidence=0.85, mahalanobis_ood_threshold=3.5),
            "brain_ct": ModalityThresholds(high_confidence=0.86, mahalanobis_ood_threshold=3.4),
            "histopathology": ModalityThresholds(high_confidence=0.90, mahalanobis_ood_threshold=3.0),
            "retina": ModalityThresholds(high_confidence=0.85, mahalanobis_ood_threshold=3.5),
            "skin": ModalityThresholds(high_confidence=0.82, mahalanobis_ood_threshold=3.8),
            "ultrasound": ModalityThresholds(high_confidence=0.80, mahalanobis_ood_threshold=4.0),
        }
    )
    covariance_regularization: float = 1e-4

    def get_thresholds_for_modality(self, modality: str) -> ModalityThresholds:
        return self.modality_thresholds.get(modality, self.default_thresholds)


__all__ = ["StrategyConfiguration", "ModalityThresholds"]
