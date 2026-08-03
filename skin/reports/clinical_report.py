"""
skin/reports/clinical_report.py
================================
Clinical report generator for skin lesion cancer predictions.
Produces structured skin_report.json diagnostic reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from skin.inference.predict_skin import SkinPredictionResult


class ClinicalReportGenerator:
    """Generates structured JSON clinical reports from skin prediction results."""

    @staticmethod
    def generate_report(
        prediction: "SkinPredictionResult",
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Serialize a prediction result to a structured clinical JSON report
        and write it to disk.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "module": "skin",
            "modality": "skin",
            "report_type": "skin_lesion_cancer_classification",
            "primary_diagnosis": {
                "predicted_class": prediction.predicted_class,
                "confidence": round(prediction.confidence, 6),
                "prediction_reliability": prediction.prediction_reliability,
                "requires_manual_review": prediction.requires_manual_review,
                "clinical_note": ClinicalReportGenerator._clinical_note(
                    prediction.predicted_class, prediction.prediction_reliability
                ),
            },
            "differential_diagnoses": [
                {"class": cls, "probability": round(prob, 6)}
                for cls, prob in prediction.top_k_predictions
            ],
            "probability_distribution": {
                cls: round(prob, 6) for cls, prob in prediction.probability_distribution.items()
            },
            "model_metadata": {
                "model_version": prediction.model_version,
                "inference_time_ms": round(prediction.inference_time_ms, 3),
                "prediction_timestamp": prediction.prediction_timestamp,
            },
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Clinical report saved to '{output_path}'.")
        return report

    @staticmethod
    def _clinical_note(predicted_class: str, reliability: str) -> str:
        """Return a brief clinical note based on predicted skin disease and reliability."""
        cls_lower = predicted_class.lower()

        if "melanoma" in cls_lower:
            return (
                f"HIGH RISK FINDING: Malignant Melanoma predicted with [{reliability}] reliability. "
                "Urgent dermatological evaluation, dermoscopy, and excisional biopsy recommended."
            )
        elif "basal_cell" in cls_lower or "bcc" in cls_lower:
            return (
                f"MODERATE RISK FINDING: Basal Cell Carcinoma (BCC) predicted with [{reliability}] reliability. "
                "Recommend dermatologic evaluation for local treatment or surgical excision."
            )
        elif "actinic_keratosis" in cls_lower or "akiec" in cls_lower or "squamous" in cls_lower:
            return (
                f"PRE-MALIGNANT / SCC FINDING: Actinic Keratosis / Intraepithelial Carcinoma predicted with [{reliability}] reliability. "
                "Recommend evaluation for cryotherapy, topical therapy, or biopsy."
            )
        elif "nevus" in cls_lower or "benign" in cls_lower:
            return (
                f"BENIGN FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Routine clinical monitoring; reassess if lesion changes in size, shape, or color."
            )
        elif "vascular" in cls_lower or "dermatofibroma" in cls_lower:
            return (
                f"BENIGN LESION FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Benign skin finding. Clinical correlation recommended if symptomatic."
            )
        else:
            return (
                f"SKIN LESION FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Clinical correlation and dermatologic examination recommended."
            )
