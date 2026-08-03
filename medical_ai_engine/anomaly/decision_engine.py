"""
decision_engine.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Final anomaly decision and clinical recommendation builder.

Architecture Role
-----------------
The DecisionEngine is the final stage of the anomaly detection pipeline (Stage 5).
It synthesizes all prior analysis results from AnomalyContext into a definitive
AnomalyStatus and RecommendedAction, producing the immutable AnomalyResult.

Decision Priority Order
-----------------------
1. POOR_QUALITY        — Image quality check fails first (useless data → discard).
2. OUT_OF_DISTRIBUTION — Image is not within model's training domain.
3. UNKNOWN_DISEASE     — Novel pathology pattern detected.
4. LOW_CONFIDENCE      — Confidence below acceptable clinical threshold.
5. KNOWN_DISEASE       — All checks pass; prediction is trusted.

ARCHITECTURE ONLY — No algorithm implemented.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_result import (
    AnomalyResult,
    AnomalyStatus,
    RecommendedAction,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class DecisionEngine:
    """
    Synthesizes all analysis signals from AnomalyContext into a final AnomalyResult.

    Recommended Action Mapping
    ---------------------------
    POOR_QUALITY        → REPEAT_ACQUISITION
    OUT_OF_DISTRIBUTION → ESCALATE
    UNKNOWN_DISEASE     → ESCALATE
    LOW_CONFIDENCE
      risk CRITICAL     → ESCALATE
      risk HIGH         → MANUAL_REVIEW
      risk MODERATE     → MANUAL_REVIEW
      risk LOW          → PROCEED (with caution)
    KNOWN_DISEASE       → PROCEED
    """

    def __init__(self, config: AnomalyConfiguration) -> None:
        self._config = config

    def decide(self, context: AnomalyContext, total_elapsed_ms: float) -> AnomalyResult:
        """
        Produce the final immutable AnomalyResult from the fully-populated AnomalyContext.

        Parameters
        ----------
        context : AnomalyContext
            Fully-populated execution context after all pipeline stages.
        total_elapsed_ms : float
            Total anomaly detection pipeline execution time in milliseconds.

        Returns
        -------
        AnomalyResult
            Immutable standardized output with full decision fields.
        """
        t0 = time.perf_counter()

        try:
            status, action, reason, known_unknown = self._determine_status(context)

            context.anomaly_status = status.value
            context.known_unknown_status = known_unknown
            context.recommended_action = action.value
            context.decision_reason = reason
            context.detection_method = self._determine_detection_method(context)

            result = AnomalyResult(
                request_id=context.request_id,
                session_id=context.session_id,
                status=status,
                risk_level=RiskLevel(context.risk_level) if context.risk_level in RiskLevel._value2member_map_ else RiskLevel.UNKNOWN,
                known_unknown_status=known_unknown,
                confidence_score=round(context.confidence, 4),
                risk_score=round(context.risk_score, 4),
                recommended_action=action,
                reason=reason,
                detection_method=context.detection_method,
                processing_time_ms=round(total_elapsed_ms, 3),
                risk_factors=list(context.risk_factors),
                confidence_analysis=context.confidence_analysis,
                ood_analysis=context.ood_analysis,
                unknown_disease_analysis=context.unknown_disease_analysis,
                warnings=list(context.warnings),
                errors=list(context.errors),
                metadata={
                    "modality": context.modality,
                    "predicted_class": context.predicted_class,
                    "quality_score": context.quality_score,
                    "performance_metrics": dict(context.performance_metrics),
                },
            )

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            context.record_metric("decision_engine_ms", elapsed_ms)
            context.log(
                f"Decision: status={status.value}, action={action.value}, "
                f"reason='{reason[:60]}...' ({elapsed_ms:.2f} ms)",
                stage="DecisionEngine",
            )
            return result

        except Exception as exc:
            # Failure recovery: return minimal NOT_ANALYZED result
            logger.exception("[DecisionEngine] Decision failed: %s", exc)
            context.add_error("DecisionEngine", str(exc))
            return AnomalyResult(
                request_id=context.request_id,
                session_id=context.session_id,
                status=AnomalyStatus.NOT_ANALYZED,
                risk_level=RiskLevel.UNKNOWN,
                known_unknown_status="UNCERTAIN",
                confidence_score=context.confidence,
                recommended_action=RecommendedAction.MANUAL_REVIEW,
                reason=f"Decision engine failure: {exc}",
                detection_method="error",
                processing_time_ms=round(total_elapsed_ms, 3),
                errors=[{"stage": "DecisionEngine", "error": str(exc)}],
            )

    def _determine_status(self, context: AnomalyContext):
        """Apply priority-ordered decision logic."""

        # Priority 1: Poor quality
        quality = context.quality_score
        if quality is not None and quality < self._config.quality_score_threshold:
            return (
                AnomalyStatus.POOR_QUALITY,
                RecommendedAction.REPEAT_ACQUISITION,
                f"Image quality score ({quality:.2f}) is below minimum threshold "
                f"({self._config.quality_score_threshold:.2f}). A new acquisition is required.",
                "UNCERTAIN",
            )

        # Priority 2: OOD
        ood_data = context.ood_analysis or {}
        if ood_data.get("is_ood", False):
            return (
                AnomalyStatus.OUT_OF_DISTRIBUTION,
                RecommendedAction.ESCALATE,
                f"Input image is detected as Out-of-Distribution (OOD score: "
                f"{ood_data.get('ood_score', 0.0):.3f}). "
                f"The image may not belong to a supported medical imaging modality.",
                "UNKNOWN",
            )

        # Priority 3: Unknown disease
        unknown_data = context.unknown_disease_analysis or {}
        if unknown_data.get("is_unknown", False):
            return (
                AnomalyStatus.UNKNOWN_DISEASE,
                RecommendedAction.ESCALATE,
                f"Detected features do not match any known disease pattern "
                f"(reconstruction_error: {unknown_data.get('reconstruction_error', 0.0):.4f}). "
                f"Possible novel pathology. Specialist review required.",
                "UNKNOWN",
            )

        # Priority 4: Low confidence
        conf_data = context.confidence_analysis or {}
        conf_flag = conf_data.get("confidence_flag", "TRUSTED")
        if conf_flag in ("LOW_CONFIDENCE", "CRITICAL"):
            risk_level = context.risk_level
            if risk_level in ("CRITICAL", "HIGH"):
                action = RecommendedAction.ESCALATE
            else:
                action = RecommendedAction.MANUAL_REVIEW

            return (
                AnomalyStatus.LOW_CONFIDENCE,
                action,
                f"Prediction confidence ({context.confidence:.2f}) is below the acceptable "
                f"clinical threshold ({self._config.confidence_threshold_low:.2f}). "
                f"Clinical risk level: {risk_level}.",
                "UNCERTAIN",
            )

        # Priority 5: Known disease — all checks passed
        return (
            AnomalyStatus.KNOWN_DISEASE,
            RecommendedAction.PROCEED,
            f"Prediction is trusted. Confidence: {context.confidence:.2f}, "
            f"Risk level: {context.risk_level}. No anomalous patterns detected.",
            "KNOWN",
        )

    @staticmethod
    def _determine_detection_method(context: AnomalyContext) -> str:
        """Resolve the primary detection method from populated analyses."""
        ood = context.ood_analysis or {}
        unk = context.unknown_disease_analysis or {}

        ood_method = ood.get("strategy_used", "")
        unk_method = unk.get("strategy_used", "")

        if ood.get("is_ood"):
            return ood_method or "ood_detector"
        if unk.get("is_unknown"):
            return unk_method or "unknown_disease_detector"
        if ood_method and ood_method != "none":
            return ood_method
        if unk_method and unk_method != "none":
            return unk_method
        return "confidence_threshold"


__all__ = ["DecisionEngine"]
