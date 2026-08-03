"""
Medical AI Engine — Clinical Report Aggregator

Aggregates disease prediction, confidence, top-K predictions, clinical notes,
severity, and recommendations into a standardized clinical report dict.

This aggregator is modality-agnostic — it reads standard keys that all
disease module adapters are expected to produce.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from medical_ai_engine.response.response_configuration import ResponseConfiguration

logger = logging.getLogger("ClinicalReportAggregator")


class ClinicalReportAggregator:
    """
    Aggregates all available prediction signals into a normalized clinical report.

    Modality-agnostic: uses only standard field names expected from all adapters
    without any modality-specific conditional logic.
    """

    def __init__(self, config: Optional[ResponseConfiguration] = None) -> None:
        self.config = config or ResponseConfiguration.default()

    def aggregate(
        self,
        modality: str,
        disease_prediction: Dict[str, Any],
        confidence_data: Dict[str, Any],
        existing_report: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Build a complete clinical report from pipeline outputs.

        Args:
            modality: Detected modality string (e.g. 'chest_xray').
            disease_prediction: Raw adapter predict() output dict.
            confidence_data: Normalised confidence dict from ConfidenceAssessmentStage.
            existing_report: Optional existing partial report from ClinicalReportHookStage.

        Returns:
            Fully aggregated clinical report dictionary.
        """
        primary_dx = disease_prediction.get("predicted_class", "unknown")
        confidence_score = float(disease_prediction.get("confidence", 0.0))
        reliability = confidence_data.get("reliability", "unknown")
        requires_review = bool(confidence_data.get("requires_manual_review", False))
        severity_grade = confidence_data.get("severity_grade")

        # Top-K differential diagnoses
        top_k: List[Dict[str, Any]] = []
        if self.config.include_differential_diagnoses:
            raw_topk = disease_prediction.get("top_k_predictions", [])
            top_k = raw_topk[: self.config.max_differential_count]

        # Clinical note
        clinical_note = self._build_clinical_note(
            modality=modality,
            primary_dx=primary_dx,
            confidence_score=confidence_score,
            reliability=reliability,
            requires_review=requires_review,
        )

        # Merge with existing partial report (from ClinicalReportHookStage)
        base = dict(existing_report) if existing_report else {}

        report: Dict[str, Any] = {
            "report_type": base.get(
                "report_type",
                f"{modality.replace('_', ' ').title()} Diagnostic Report",
            ),
            "schema_version": "1.0.0",
            "primary_diagnosis": primary_dx,
            "confidence_score": round(confidence_score, 6),
            "reliability": reliability,
            "requires_manual_review": requires_review,
            "differential_diagnoses": top_k,
            "clinical_note": clinical_note if self.config.include_clinical_note else "",
        }

        if self.config.include_severity_grade and severity_grade is not None:
            report["severity_grade"] = severity_grade

        # Recommendations based on reliability
        report["recommendations"] = self._build_recommendations(
            reliability=reliability,
            requires_review=requires_review,
            primary_dx=primary_dx,
        )

        return report

    def _build_clinical_note(
        self,
        modality: str,
        primary_dx: str,
        confidence_score: float,
        reliability: str,
        requires_review: bool,
    ) -> str:
        """Generate an auto-populated clinical note string."""
        review_note = " Manual clinician review is recommended." if requires_review else ""
        return (
            f"AI-assisted diagnostic analysis performed on '{modality}' image. "
            f"Primary finding: '{primary_dx}' with confidence {confidence_score:.4f} "
            f"({reliability} reliability).{review_note}"
        )

    def _build_recommendations(
        self,
        reliability: str,
        requires_review: bool,
        primary_dx: str,
    ) -> List[str]:
        """Derive clinical recommendations from reliability and prediction data."""
        recs: List[str] = []

        if reliability == "high" and not requires_review:
            recs.append(f"Primary finding '{primary_dx}' confirmed with high confidence. Proceed per clinical protocol.")
        elif reliability == "moderate":
            recs.append(f"Moderate confidence in '{primary_dx}'. Consider correlation with clinical history.")
        else:
            recs.append(f"Low confidence prediction for '{primary_dx}'. Manual clinical review strongly advised.")

        if requires_review:
            recs.append("This case has been flagged for mandatory clinical review before action.")

        return recs
