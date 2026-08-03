"""
risk_assessor.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Multi-signal clinical risk level evaluation.

Architecture Role
-----------------
The RiskAssessor is Stage 4 of the anomaly detection pipeline.
It fuses all available signals (confidence, OOD score, reconstruction error, quality score)
into a composite risk score and assigns a clinical risk level.

Clinical Risk Levels
---------------------
  LOW       → Prediction is reliable; proceed with standard workflow.
  MODERATE  → Slight uncertainty; clinician should be aware.
  HIGH      → Significant uncertainty; manual review strongly recommended.
  CRITICAL  → Extremely low confidence or severe anomaly detected; escalate immediately.

ARCHITECTURE ONLY — Scoring weights and algorithm implemented in Module 16.2+.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_result import RiskLevel

logger = logging.getLogger(__name__)


class RiskAssessor:
    """
    Multi-signal clinical risk level evaluator.

    Fuses:
      - Confidence signal (from ConfidenceAnalyzer)
      - OOD signal (from OODDetector)
      - Unknown disease signal (from UnknownDiseaseDetector)
      - Image quality score (from PredictionResult/EngineContext)

    Populates:
      context.risk_level    → RiskLevel enum string
      context.risk_score    → Composite float (0.0–1.0)
      context.risk_factors  → Human-readable list of contributing factors
    """

    # Architecture-level risk weight schema (finalized in Module 16.2)
    _CONFIDENCE_WEIGHT: float = 0.40
    _OOD_WEIGHT: float = 0.30
    _UNKNOWN_DISEASE_WEIGHT: float = 0.20
    _QUALITY_WEIGHT: float = 0.10

    def __init__(self, config: AnomalyConfiguration) -> None:
        self._config = config

    def assess(self, context: AnomalyContext) -> None:
        """
        Compute composite risk score and assign risk level.
        Populates context.risk_level, risk_score, and risk_factors.
        """
        t0 = time.perf_counter()

        try:
            risk_score, risk_factors = self._compute_risk_score(context)
            risk_level = self._classify_risk_level(risk_score)

            context.risk_score = risk_score
            context.risk_level = risk_level.value
            context.risk_factors = risk_factors

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.record_metric("risk_assessment_ms", elapsed_ms)
            context.log(
                f"Risk assessment: level={risk_level.value}, score={risk_score:.4f}, "
                f"factors={len(risk_factors)} ({elapsed_ms:.2f} ms)",
                stage="RiskAssessor",
            )

        except Exception as exc:
            context.add_error("RiskAssessor", str(exc))
            context.risk_level = RiskLevel.UNKNOWN.value
            context.risk_score = 0.0
            logger.exception("[RiskAssessor] Risk assessment failed: %s", exc)

    def _compute_risk_score(self, context: AnomalyContext) -> Tuple[float, List[str]]:
        """Compute weighted composite risk score [0.0, 1.0]."""
        risk_score = 0.0
        risk_factors: List[str] = []

        # 1. Confidence signal contribution
        conf_risk = 1.0 - context.confidence
        risk_score += conf_risk * self._CONFIDENCE_WEIGHT
        if context.confidence < self._config.confidence_threshold_critical:
            risk_factors.append(
                f"Critical confidence ({context.confidence:.2f} < {self._config.confidence_threshold_critical:.2f})"
            )
        elif context.confidence < self._config.confidence_threshold_low:
            risk_factors.append(
                f"Low confidence ({context.confidence:.2f} < {self._config.confidence_threshold_low:.2f})"
            )

        # 2. OOD signal contribution
        ood_data = context.ood_analysis or {}
        is_ood = ood_data.get("is_ood", False)
        ood_score_raw = ood_data.get("ood_score", 0.0)
        if is_ood:
            risk_score += 1.0 * self._OOD_WEIGHT
            risk_factors.append(f"OOD detected (score={ood_score_raw:.3f})")
        else:
            ood_contrib = min(ood_score_raw / max(self._config.ood_distance_threshold, 1e-6), 1.0)
            risk_score += ood_contrib * self._OOD_WEIGHT

        # 3. Unknown disease signal contribution
        unknown_data = context.unknown_disease_analysis or {}
        is_unknown = unknown_data.get("is_unknown", False)
        rec_error = unknown_data.get("reconstruction_error") or 0.0
        if is_unknown:
            risk_score += 1.0 * self._UNKNOWN_DISEASE_WEIGHT
            risk_factors.append(f"Unknown disease pattern (reconstruction_error={rec_error:.4f})")
        else:
            unk_contrib = min(rec_error / max(self._config.reconstruction_error_threshold, 1e-6), 1.0)
            risk_score += unk_contrib * self._UNKNOWN_DISEASE_WEIGHT

        # 4. Image quality signal contribution
        quality = context.quality_score
        if quality is not None:
            quality_risk = 1.0 - quality
            risk_score += quality_risk * self._QUALITY_WEIGHT
            if quality < self._config.quality_score_threshold:
                risk_factors.append(f"Poor image quality (score={quality:.2f})")

        risk_score = min(max(round(risk_score, 4), 0.0), 1.0)
        return risk_score, risk_factors

    @staticmethod
    def _classify_risk_level(risk_score: float) -> RiskLevel:
        """Map composite risk score to clinical RiskLevel."""
        if risk_score >= 0.80:
            return RiskLevel.CRITICAL
        if risk_score >= 0.55:
            return RiskLevel.HIGH
        if risk_score >= 0.30:
            return RiskLevel.MODERATE
        return RiskLevel.LOW


__all__ = ["RiskAssessor"]
