"""
risk_assessor.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Clinical risk assessment and risk factor accumulation.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from medical_ai_engine.anomaly.anomaly_result import AnomalyStatus, RiskLevel
from .decision_configuration import DecisionConfiguration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskAssessmentOutput:
    """
    Structured outcome of clinical risk assessment.

    Attributes
    ----------
    risk_level : RiskLevel
        Clinical risk classification (LOW, MEDIUM/MODERATE, HIGH, CRITICAL).
    overall_risk_score : float
        Normalized numeric score in [0.0, 1.0].
    risk_factors : List[str]
        Human-readable summary of risk drivers.
    """
    risk_level: RiskLevel
    overall_risk_score: float
    risk_factors: List[str]

    def as_dict(self) -> Dict[str, Any]:
        return {
            "risk_level": self.risk_level.value,
            "overall_risk_score": round(self.overall_risk_score, 4),
            "risk_factors": self.risk_factors,
        }


class RiskAssessor:
    """
    Assesses clinical risk levels from composite risk scores and fused decision outputs.

    Risk Levels Supported
    ---------------------
    - **LOW**: In-distribution prediction with high confidence.
    - **MEDIUM** / **MODERATE**: Minor confidence degradation or borderline metrics.
    - **HIGH**: Out-of-Distribution or Unknown disease flag raised.
    - **CRITICAL**: Extreme OOD distance or critical quality failure.
    """

    def __init__(self, config: Optional[DecisionConfiguration] = None) -> None:
        self._cfg = config or DecisionConfiguration.default()

    def assess(
        self,
        status: AnomalyStatus,
        risk_score: float,
        confidence: float,
        signals: Dict[str, Any],
        modality: str = "chest_xray",
    ) -> RiskAssessmentOutput:
        """
        Compute final clinical risk level and risk factors.

        Parameters
        ----------
        status : AnomalyStatus
            Fused anomaly status.
        risk_score : float
            Numeric risk score in [0.0, 1.0].
        confidence : float
            Model prediction confidence.
        signals : Dict[str, Any]
            Map of signal details.
        modality : str
            Imaging modality name.

        Returns
        -------
        RiskAssessmentOutput
        """
        factors: List[str] = []
        score = max(0.0, min(1.0, float(risk_score)))

        th_high = self._cfg.per_modality_thresholds.get(modality, self._cfg.high_risk_threshold)
        th_crit = self._cfg.critical_risk_threshold

        # Accumulate risk factors
        if status == AnomalyStatus.UNKNOWN_DISEASE:
            factors.append("Unknown disease pattern detected by prototype embedding comparison.")
        if status == AnomalyStatus.OUT_OF_DISTRIBUTION:
            factors.append("Sample lies outside training feature distribution.")
        if status == AnomalyStatus.LOW_CONFIDENCE:
            factors.append(f"Low prediction confidence ({confidence:.2f}).")
        if status == AnomalyStatus.POOR_QUALITY:
            factors.append("Sub-standard image acquisition quality.")

        # Determine level
        if score >= th_crit or status in (AnomalyStatus.UNKNOWN_DISEASE, AnomalyStatus.OUT_OF_DISTRIBUTION) and score > 0.85:
            level = RiskLevel.CRITICAL if score >= th_crit else RiskLevel.HIGH
        elif score >= th_high or status in (AnomalyStatus.UNKNOWN_DISEASE, AnomalyStatus.OUT_OF_DISTRIBUTION):
            level = RiskLevel.HIGH
        elif score >= 0.35 or status == AnomalyStatus.LOW_CONFIDENCE:
            level = RiskLevel.MODERATE
        else:
            level = RiskLevel.LOW
            if not factors:
                factors.append("In-distribution high-confidence prediction.")

        return RiskAssessmentOutput(
            risk_level=level,
            overall_risk_score=score,
            risk_factors=factors,
        )


__all__ = ["RiskAssessor", "RiskAssessmentOutput"]
