"""
brain_mri/reports/clinical_report.py
====================================
Generates structured clinical JSON reports per prediction.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class ClinicalReportData:
    predicted_disease: str
    confidence: float
    probability_distribution: Dict[str, float]
    top_3_predictions: List[Dict[str, Any]]
    model_version: str
    inference_time_ms: float
    prediction_timestamp: str
    modality: str = "brain_mri"
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ClinicalReportGenerator:
    """
    Generates and saves structured clinical diagnostic reports.
    """

    def __init__(self, reports_dir: str | Path = "saved_models/brain_mri/reports", model_version: str = "0.7.0"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.model_version = model_version

    def generate_report(
        self,
        predicted_disease: str,
        confidence: float,
        probability_distribution: Dict[str, float],
        top_3_predictions: List[Dict[str, Any]],
        inference_time_ms: float,
        image_path: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> ClinicalReportData:
        """
        Creates ClinicalReportData and writes brain_mri_report.json file.
        """
        now_str = datetime.datetime.now(datetime.timezone.utc).isoformat()

        report_data = ClinicalReportData(
            predicted_disease=predicted_disease,
            confidence=round(confidence, 4),
            probability_distribution={k: round(v, 4) for k, v in probability_distribution.items()},
            top_3_predictions=top_3_predictions,
            model_version=self.model_version,
            inference_time_ms=round(inference_time_ms, 2),
            prediction_timestamp=now_str,
            image_path=image_path,
        )

        filename = output_filename or "brain_mri_report.json"
        save_path = self.reports_dir / filename
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(report_data.to_dict(), f, indent=2)

        logger.info(f"Generated clinical diagnostic report: {save_path}")
        return report_data
