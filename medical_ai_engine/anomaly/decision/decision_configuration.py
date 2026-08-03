"""
decision_configuration.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Configuration dataclass for Decision Fusion & Risk Assessment.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from .decision_exceptions import DecisionConfigurationError


@dataclass
class DecisionConfiguration:
    """
    Configuration for the Decision Fusion and Risk Assessment subsystem.

    Attributes
    ----------
    policy_name : str
        Active fusion policy ("default_priority" | "weighted_fusion").
    priority_order : List[str]
        Order of signal evaluation priority. Default: unknown_disease > ood > low_confidence > poor_quality.
    confidence_weight : float
        Weight assigned to confidence signal in weighted fusion policy.
    ood_weight : float
        Weight assigned to OOD signal in weighted fusion policy.
    unknown_weight : float
        Weight assigned to unknown disease signal in weighted fusion policy.
    high_risk_threshold : float
        Risk score threshold for HIGH risk level classification.
    critical_risk_threshold : float
        Risk score threshold for CRITICAL risk level classification.
    per_modality_thresholds : Dict[str, float]
        Optional per-modality risk score threshold overrides.
    """
    policy_name: str = "default_priority"
    priority_order: List[str] = field(
        default_factory=lambda: [
            "unknown_disease",
            "ood",
            "low_confidence",
            "poor_quality",
        ]
    )
    confidence_weight: float = 0.30
    ood_weight: float = 0.30
    unknown_weight: float = 0.40
    high_risk_threshold: float = 0.70
    critical_risk_threshold: float = 0.88
    per_modality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "brain_mri": 0.68,
            "chest_xray": 0.70,
            "histopathology": 0.65,
            "retina": 0.70,
            "skin": 0.72,
            "ultrasound": 0.75,
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.high_risk_threshold < 1.0:
            raise DecisionConfigurationError(
                f"high_risk_threshold must be in (0,1); got {self.high_risk_threshold}"
            )
        if not 0.0 < self.critical_risk_threshold <= 1.0:
            raise DecisionConfigurationError(
                f"critical_risk_threshold must be in (0,1]; got {self.critical_risk_threshold}"
            )
        if self.high_risk_threshold >= self.critical_risk_threshold:
            raise DecisionConfigurationError("high_risk_threshold must be < critical_risk_threshold")

    @classmethod
    def default(cls) -> "DecisionConfiguration":
        return cls()

    @classmethod
    def weighted(cls) -> "DecisionConfiguration":
        return cls(policy_name="weighted_fusion")


__all__ = ["DecisionConfiguration"]
