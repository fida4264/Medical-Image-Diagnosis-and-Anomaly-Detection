"""
ultrasound/reports/clinical_report.py
======================================
Clinical report generator for ultrasound disease predictions.
Produces structured ultrasound_report.json diagnostic reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ultrasound.inference.predict_ultrasound import UltrasoundPredictionResult


class ClinicalReportGenerator:
    """Generates structured JSON clinical reports from ultrasound prediction results."""

    @staticmethod
    def generate_report(
        prediction: "UltrasoundPredictionResult",
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Serialize a prediction result to a structured clinical JSON report
        and write it to disk.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "module": "ultrasound",
            "modality": "ultrasound",
            "report_type": "ultrasound_disease_diagnosis",
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
        """Return a brief clinical note based on predicted ultrasound disease and reliability."""
        cls_lower = predicted_class.lower()

        if "malignant" in cls_lower:
            return (
                f"HIGH RISK FINDING: Malignant lesion predicted with [{reliability}] reliability. "
                "Immediate specialist referral, BI-RADS classification, and histological biopsy recommended."
            )
        elif "benign" in cls_lower:
            return (
                f"BENIGN FINDING: Benign ultrasound lesion predicted with [{reliability}] reliability. "
                "Routine clinical monitoring and follow-up sonogram recommended as indicated."
            )
        elif "normal" in cls_lower:
            return (
                f"NORMAL FINDING: Unremarkable ultrasound study predicted with [{reliability}] reliability. "
                "No focal mass or sonographic anomaly identified."
            )
        elif "stone" in cls_lower or "calculus" in cls_lower:
            return (
                f"CALCULUS FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Ultrasonographic acoustic shadowing noted. Clinical and urological correlation recommended."
            )
        elif "cyst" in cls_lower:
            return (
                f"CYSTIC FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Anecotic lesion consistent with cyst. Follow-up imaging recommended if symptomatic."
            )
        else:
            return (
                f"ULTRASOUND FINDING: {predicted_class} predicted with [{reliability}] reliability. "
                "Clinical correlation and targeted diagnostic evaluation recommended."
            )
