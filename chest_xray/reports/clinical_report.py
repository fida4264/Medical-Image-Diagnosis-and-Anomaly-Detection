"""
chest_xray/reports/clinical_report.py
=======================================
Generates chest_xray_report.json and standardized prediction JSON output.
Mirrors BrainMRI ClinicalReportGenerator.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

from chest_xray.inference.predict_chest_xray import ChestXrayPredictionResult

logger = logging.getLogger(__name__)


class ClinicalReportGenerator:
    """Generates clinical JSON report (chest_xray_report.json) for clinical workflows."""

    @staticmethod
    def generate_report(
        result: ChestXrayPredictionResult,
        output_path: str | Path = "saved_models/chest_xray/reports/chest_xray_report.json",
    ) -> Tuple[Dict[str, Any], Path]:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        report = {
            "report_title": "Chest X-ray Disease Diagnosis & Clinical Findings",
            "modality": "chest_xray",
            "clinical_finding": {
                "primary_diagnosis": result.predicted_class,
                "diagnostic_confidence": f"{result.confidence * 100:.2f}%",
                "confidence_score": result.confidence,
            },
            "differential_diagnoses": [
                {
                    "rank": pred.rank,
                    "disease": pred.disease,
                    "probability": f"{pred.confidence * 100:.2f}%",
                }
                for pred in result.top3_predictions
            ],
            "full_probability_distribution": result.probabilities,
            "metadata": {
                "model_version": result.model_version,
                "inference_latency_ms": result.inference_time_ms,
                "timestamp_utc": result.timestamp,
                "source_image": result.image_path,
                "status": result.status,
            },
            "standardized_output": result.to_standard_dict(),
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Generated clinical report at {out_p}")
        return report, out_p
