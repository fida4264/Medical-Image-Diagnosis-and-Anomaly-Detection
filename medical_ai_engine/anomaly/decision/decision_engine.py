"""
decision_engine.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Master Decision Engine orchestrating signal fusion into a single unified AnomalyResult.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Optional

from medical_ai_engine.anomaly.anomaly_result import AnomalyResult, AnomalyStatus, RecommendedAction, RiskLevel
from .decision_configuration import DecisionConfiguration
from .decision_exceptions import DecisionError
from .fusion_engine import FusionEngine
from .priority_manager import PriorityManager
from .risk_assessor import RiskAssessor

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Master Decision Engine for the Anomaly Detection subsystem.

    Combines input signals:
    - Confidence Threshold Result
    - Mahalanobis OOD Result
    - Unknown Disease Detection Result
    - Image Quality Result

    Produces a single, immutable AnomalyResult containing:
    - Final Risk Level (LOW, MEDIUM/MODERATE, HIGH, CRITICAL)
    - Overall Risk Score (0.0 - 1.0)
    - Known/Unknown Status ("KNOWN", "UNKNOWN", "UNCERTAIN")
    - Recommended Action (PROCEED, MANUAL_REVIEW, REPEAT_SCAN, ESCALATE, REJECT_PREDICTION)
    - Decision Reason
    - Decision Confidence
    """

    def __init__(
        self,
        config: Optional[DecisionConfiguration] = None,
        fusion_engine: Optional[FusionEngine] = None,
        risk_assessor: Optional[RiskAssessor] = None,
    ) -> None:
        self._cfg = config or DecisionConfiguration.default()
        self._fusion = fusion_engine or FusionEngine(config=self._cfg)
        self._assessor = risk_assessor or RiskAssessor(config=self._cfg)
        self._lock = threading.Lock()
        logger.info("DecisionEngine (Module 16.5) initialised.")

    def evaluate_signals(
        self,
        prediction_result: Any,
        confidence_signal: Optional[Dict[str, Any]] = None,
        ood_signal: Optional[Dict[str, Any]] = None,
        unknown_signal: Optional[Dict[str, Any]] = None,
        quality_signal: Optional[Dict[str, Any]] = None,
    ) -> AnomalyResult:
        """
        Synthesize all detection signals into a single unified AnomalyResult.

        Parameters
        ----------
        prediction_result : Any
            Prediction object exposing request_id, session_id, modality, confidence.
        confidence_signal : Dict[str, Any] or None
            Signal from confidence strategy.
        ood_signal : Dict[str, Any] or None
            Signal from Mahalanobis OOD strategy.
        unknown_signal : Dict[str, Any] or None
            Signal from Unknown Disease detector.
        quality_signal : Dict[str, Any] or None
            Signal from image quality check.

        Returns
        -------
        AnomalyResult
        """
        t0 = time.perf_counter()
        try:
            req_id = str(getattr(prediction_result, "request_id", "req-unknown"))
            sess_id = str(getattr(prediction_result, "session_id", ""))
            modality = str(getattr(prediction_result, "modality", "chest_xray"))
            confidence = float(getattr(prediction_result, "confidence", 0.0))

            # 1. Fuse signals via active policy
            fused_output = self._fusion.fuse(
                confidence_signal=confidence_signal,
                ood_signal=ood_signal,
                unknown_signal=unknown_signal,
                quality_signal=quality_signal,
                confidence=confidence,
                modality=modality,
            )

            # 2. Perform clinical risk assessment
            signals_map = {
                "low_confidence": confidence_signal or {},
                "ood": ood_signal or {},
                "unknown_disease": unknown_signal or {},
                "poor_quality": quality_signal or {},
            }
            risk_assessment = self._assessor.assess(
                status=fused_output.status,
                risk_score=fused_output.risk_score,
                confidence=confidence,
                signals=signals_map,
                modality=modality,
            )

            processing_time = (time.perf_counter() - t0) * 1000.0

            result = AnomalyResult(
                request_id=req_id,
                session_id=sess_id,
                status=fused_output.status,
                risk_level=risk_assessment.risk_level,
                known_unknown_status=fused_output.known_unknown_status,
                confidence_score=confidence,
                risk_score=risk_assessment.overall_risk_score,
                recommended_action=fused_output.recommended_action,
                reason=fused_output.reason,
                detection_method=f"decision_fusion_{fused_output.primary_signal}",
                processing_time_ms=processing_time,
                risk_factors=risk_assessment.risk_factors,
                confidence_analysis=confidence_signal,
                ood_analysis=ood_signal,
                unknown_disease_analysis=unknown_signal,
                metadata={
                    "decision_confidence": fused_output.decision_confidence,
                    "primary_signal": fused_output.primary_signal,
                    "policy_name": self._fusion.current_policy_name,
                },
            )

            logger.info(
                "DecisionEngine [%s]: status=%s risk=%s action=%s (score=%.3f)",
                req_id, result.status.value, result.risk_level.value, result.recommended_action.value, result.risk_score,
            )
            return result

        except Exception as exc:
            raise DecisionError(f"DecisionEngine evaluation failed: {exc}") from exc

    def health_check(self) -> Dict[str, Any]:
        return {
            "component": "DecisionEngine",
            "module": "16.5",
            "status": "healthy",
            "policy": self._fusion.current_policy_name,
        }


__all__ = ["DecisionEngine"]
