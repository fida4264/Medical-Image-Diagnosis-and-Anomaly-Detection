"""
recommendation_builder.py — Module 18: Medical Report Generation Engine
Medical AI Platform — Translates predictions and anomaly signals into clinical recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClinicalRecommendation:
    """
    Formulated clinical recommendation and action guidance.

    Fields
    ------
    action_code : str
        Standardized action code (e.g. "URGENT_CLINICAL_REVIEW", "ROUTINE_FOLLOW_UP").
    recommended_action : str
        Human-readable recommended clinical action.
    urgency_level : str
        Urgency classification: "EMERGENT", "URGENT", "ROUTINE", "LOW_PRIORITY".
    rationale : str
        Clinical rationale explaining why this recommendation was selected.
    suggested_follow_up : str
        Specific follow-up diagnostic procedures or specialist consultations.
    """

    action_code: str
    recommended_action: str
    urgency_level: str
    rationale: str
    suggested_follow_up: str


class RecommendationBuilder:
    """
    Formulates clinical recommendations and next steps based on disease prediction,
    model confidence, and anomaly risk assessment.
    """

    def build_recommendation(
        self,
        prediction_result: Any,
        anomaly_result: Optional[Any] = None,
    ) -> ClinicalRecommendation:
        """
        Build ClinicalRecommendation object.

        Parameters
        ----------
        prediction_result : Any
            PredictionResult object or dictionary.
        anomaly_result : Any, optional
            AnomalyResult object or dictionary.

        Returns
        -------
        ClinicalRecommendation
        """
        disease, confidence = self._extract_pred(prediction_result)
        status, risk_level, rec_action = self._extract_anomaly(anomaly_result)

        # Determine recommendation logic
        if status in ("UNKNOWN_DISEASE", "OUT_OF_DISTRIBUTION") or risk_level in ("HIGH", "CRITICAL"):
            action_code = rec_action or "IMMEDIATE_CLINICAL_TRIAGE"
            recommended_action = "Immediate Clinical Triage & Specialist Referral Required"
            urgency_level = "EMERGENT" if risk_level == "CRITICAL" else "URGENT"
            rationale = (
                f"Anomaly engine flagged status={status} with risk_level={risk_level}. "
                f"Prediction of {disease} ({confidence*100:.1f}%) requires expert validation "
                f"due to potential out-of-distribution features or novel disease presentation."
            )
            suggested_follow_up = (
                "1. Immediate review by attending radiologist or domain specialist.\n"
                "2. Consider high-resolution confirmatory imaging or biopsy.\n"
                "3. Perform complete clinical history and physical examination."
            )

        elif confidence < 0.70 or status == "LOW_CONFIDENCE":
            action_code = "REPEAT_EXAMINATION_OR_SECOND_OPINION"
            recommended_action = "Repeat Examination or Secondary Expert Opinion Recommended"
            urgency_level = "URGENT"
            rationale = (
                f"Model confidence ({confidence*100:.1f}%) is below diagnostic threshold. "
                f"Image quality or subtle pathology requires secondary verification."
            )
            suggested_follow_up = (
                "1. Secondary review by senior medical officer.\n"
                "2. Check image acquisition parameters for motion or artifact pollution.\n"
                "3. Order follow-up scan if clinically indicated."
            )

        elif disease.lower() in ("normal", "no tumor", "no dr", "benign") and confidence >= 0.85:
            action_code = "ROUTINE_FOLLOW_UP"
            recommended_action = "Standard Routine Follow-up"
            urgency_level = "ROUTINE"
            rationale = (
                f"High-confidence negative finding ({disease}, {confidence*100:.1f}% confidence). "
                f"No acute abnormal diagnostic features detected."
            )
            suggested_follow_up = (
                "1. Continue standard routine screening schedule.\n"
                "2. Re-evaluate if new clinical symptoms develop."
            )

        else:
            action_code = rec_action or "CLINICAL_CORRELATION_RECOMMENDED"
            recommended_action = "Clinical Correlation & Specialist Follow-up Recommended"
            urgency_level = "URGENT" if confidence >= 0.90 else "ROUTINE"
            rationale = (
                f"Positive diagnostic detection for {disease} ({confidence*100:.1f}% confidence) "
                f"within normal distribution parameters."
            )
            suggested_follow_up = (
                "1. Correlate with patient clinical presentation and lab values.\n"
                "2. Schedule appropriate specialist consultation.\n"
                "3. Initiate standard treatment protocol if confirmed."
            )

        return ClinicalRecommendation(
            action_code=action_code,
            recommended_action=recommended_action,
            urgency_level=urgency_level,
            rationale=rationale,
            suggested_follow_up=suggested_follow_up,
        )

    def _extract_pred(self, pred: Any) -> tuple:
        if isinstance(pred, dict):
            disease = pred.get("predicted_disease") or pred.get("predicted_class") or "Unknown"
            conf = float(pred.get("confidence", 0.0))
        else:
            disease = getattr(pred, "predicted_disease", None) or getattr(pred, "predicted_class", "Unknown")
            conf = float(getattr(pred, "confidence", 0.0))
        return disease, conf

    def _extract_anomaly(self, anomaly: Optional[Any]) -> tuple:
        if anomaly is None:
            return "KNOWN_DISEASE", "LOW", None
        if isinstance(anomaly, dict):
            status = anomaly.get("status", "KNOWN_DISEASE")
            risk = anomaly.get("risk_level", "LOW")
            rec = anomaly.get("recommended_action")
        else:
            status = getattr(anomaly, "status", "KNOWN_DISEASE")
            if hasattr(status, "value"):
                status = status.value
            risk = getattr(anomaly, "risk_level", "LOW")
            if hasattr(risk, "value"):
                risk = risk.value
            rec = getattr(anomaly, "recommended_action", None)
            if hasattr(rec, "value"):
                rec = rec.value

        return str(status), str(risk), rec


__all__ = ["RecommendationBuilder", "ClinicalRecommendation"]
