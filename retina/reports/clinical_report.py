"""
retina/reports/clinical_report.py
==================================
Clinical report generator for retina disease predictions and severity grading.
Produces structured retina_report.json diagnostic reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from retina.inference.predict_retina import RetinaPredictionResult


class ClinicalReportGenerator:
    """Generates structured JSON clinical reports from retina prediction results."""

    @staticmethod
    def generate_report(
        prediction: "RetinaPredictionResult",
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Serialize a prediction result to a structured clinical JSON report
        and write it to disk.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "module": "retina",
            "modality": "retina",
            "report_type": "retinal_disease_diagnosis_and_grading",
            "primary_diagnosis": {
                "predicted_class": prediction.predicted_class,
                "severity_stage": prediction.severity,
                "confidence": round(prediction.confidence, 6),
                "clinical_note": ClinicalReportGenerator._clinical_note(
                    prediction.predicted_class, prediction.severity
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
    def _clinical_note(predicted_class: str, severity: str) -> str:
        """Return a brief clinical note based on predicted disease and severity stage."""
        cls_lower = predicted_class.lower()
        if "normal" in cls_lower:
            return "Normal retinal fundus exam. No pathological lesions detected."
        elif "diabetic" in cls_lower or "dr" in cls_lower:
            return f"Diabetic Retinopathy finding mapped to stage [{severity}]. Recommend ophthalmologist evaluation and glycemic control."
        elif "glaucoma" in cls_lower:
            return f"Glaucomatous optic neuropathy finding mapped to stage [{severity}]. Recommend intraocular pressure (IOP) measurement and visual field test."
        elif "amd" in cls_lower or "macular" in cls_lower:
            return f"Age-related Macular Degeneration (AMD) finding mapped to stage [{severity}]. Recommend OCT scanning and Amsler grid monitoring."
        elif "hypertensive" in cls_lower:
            return f"Hypertensive retinopathy finding mapped to stage [{severity}]. Recommend blood pressure control and dilated eye exam."
        else:
            return f"Retinal lesion detected ({predicted_class}) mapped to stage [{severity}]. Clinical correlation and fundus check recommended."
