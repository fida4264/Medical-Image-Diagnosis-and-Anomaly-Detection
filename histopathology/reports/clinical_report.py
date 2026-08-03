"""
histopathology/reports/clinical_report.py
==========================================
Clinical report generator for histopathology cancer subtype predictions.
Produces structured JSON diagnostic reports. Mirrors Module 7/8/9 pattern.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from histopathology.inference.predict_histopathology import HistopathologyPredictionResult


class ClinicalReportGenerator:
    """Generates structured JSON clinical reports from prediction results."""

    @staticmethod
    def generate_report(
        prediction: "HistopathologyPredictionResult",
        output_path: Path,
    ) -> Dict[str, Any]:
        """
        Serialize a prediction result to a structured clinical JSON report
        and write it to disk.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "module": "histopathology",
            "modality": "histopathology",
            "report_type": "cancer_subtype_diagnosis",
            "primary_diagnosis": {
                "predicted_class": prediction.predicted_class,
                "confidence": round(prediction.confidence, 6),
                "clinical_note": ClinicalReportGenerator._clinical_note(prediction.predicted_class),
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
    def _clinical_note(predicted_class: str) -> str:
        """Return a brief clinical note based on the predicted cancer subtype."""
        notes = {
            "colon_adenocarcinoma": "Malignant glandular epithelial tumor of the colon. Requires oncology referral.",
            "colon_normal": "Normal colon tissue. No malignant features detected.",
            "lung_adenocarcinoma": "Most common subtype of lung cancer. Peripheral location typical. Requires staging.",
            "lung_normal": "Normal lung tissue. No malignant features detected.",
            "lung_squamous_cell": "Squamous cell carcinoma of the lung. Central location typical. Requires staging.",
        }
        return notes.get(predicted_class, "Histopathological finding. Requires clinical correlation.")
