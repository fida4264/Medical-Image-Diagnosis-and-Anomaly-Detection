"""
history_manager.py — Module 19: Prediction History & Database
Medical AI Platform — High-level history manager for storing, querying, and exporting records.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from medical_ai_engine.database.database_configuration import DatabaseConfiguration
from medical_ai_engine.database.database_exceptions import ExportError, RepositoryError
from medical_ai_engine.database.database_manager import DatabaseManager
from medical_ai_engine.database.prediction_repository import PredictionRecord, PredictionRepository
from medical_ai_engine.database.query_engine import QueryEngine
from medical_ai_engine.database.statistics_engine import PlatformDatabaseStatistics, StatisticsEngine

logger = logging.getLogger(__name__)


class HistoryManager:
    """
    High-level facade for prediction history storage, querying, analytics, and CSV/JSON exports.

    Interacts cleanly with MedicalAIEngine without modifying prediction models or core pipeline logic.
    """

    def __init__(
        self,
        config: Optional[DatabaseConfiguration] = None,
        db_manager: Optional[DatabaseManager] = None,
    ) -> None:
        self.config = config or DatabaseConfiguration.default()
        self.db_manager = db_manager or DatabaseManager(self.config)
        self.repository = PredictionRepository(self.db_manager)
        self.query_engine = QueryEngine(self.db_manager)
        self.statistics_engine = StatisticsEngine(self.db_manager)

    def record_prediction(
        self,
        prediction_result: Any,
        explainability_result: Optional[Any] = None,
        anomaly_result: Optional[Any] = None,
        report_path: Optional[str] = None,
    ) -> str:
        """
        Extract diagnostic metadata and record a completed prediction cycle.

        Parameters
        ----------
        prediction_result : Any
            PredictionResult object or dictionary.
        explainability_result : Any, optional
            ExplainabilityResult object or dictionary.
        anomaly_result : Any, optional
            AnomalyResult object or dictionary.
        report_path : str, optional
            File path of generated medical report artifact.

        Returns
        -------
        str
            Prediction ID / Request ID of recorded entry.
        """
        try:
            req_id, patient_id, case_id, modality, disease, conf, lat_pred = self._extract_pred(prediction_result)
            exp_summary, heatmap_path, lat_exp = self._extract_exp(explainability_result)
            status, risk_level, risk_score, rec_action, lat_anom = self._extract_anomaly(anomaly_result)

            total_lat = lat_pred + lat_exp + lat_anom

            record = PredictionRecord(
                prediction_id=req_id,
                patient_id=patient_id,
                case_id=case_id,
                modality=modality,
                predicted_disease=disease,
                confidence=conf,
                explainability_summary=exp_summary,
                heatmap_path=heatmap_path,
                anomaly_status=status,
                risk_level=risk_level,
                risk_score=risk_score,
                recommended_action=rec_action,
                report_path=report_path or "",
                processing_time_ms=total_lat,
            )

            res_id = self.repository.save(record)
            logger.info("[HistoryManager] Successfully recorded prediction %s in database.", res_id)
            return res_id
        except Exception as exc:
            logger.error("[HistoryManager] Failed to record prediction: %s", exc)
            raise RepositoryError(f"Failed to record prediction in database: {exc}") from exc

    def get_prediction(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Fetch prediction record by ID."""
        return self.query_engine.get_by_id(prediction_id)

    def get_history(self, limit: int = 50, offset: int = 0) -> List[PredictionRecord]:
        """Fetch paginated prediction history."""
        return self.query_engine.search(limit=limit, offset=offset)

    def filter_history(self, **kwargs) -> List[PredictionRecord]:
        """Filter prediction history with custom search options."""
        return self.query_engine.search(**kwargs)

    def get_statistics(self) -> PlatformDatabaseStatistics:
        """Compute database analytics summary."""
        return self.statistics_engine.compute_statistics()

    # ------------------------------------------------------------------
    # CSV / JSON Export Functions
    # ------------------------------------------------------------------

    def export_csv(
        self,
        output_path: Union[str, Path] = "docs/sample_prediction_history.csv",
        records: Optional[List[PredictionRecord]] = None,
    ) -> Path:
        """
        Export prediction records to CSV file.

        Parameters
        ----------
        output_path : str or Path
            Target CSV file path.
        records : List[PredictionRecord], optional
            Records to export. If None, exports all records in database.

        Returns
        -------
        Path
        """
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            if records is None:
                records = self.get_history(limit=10_000)

            fieldnames = [
                "prediction_id",
                "created_at",
                "patient_id",
                "case_id",
                "modality",
                "predicted_disease",
                "confidence",
                "explainability_summary",
                "heatmap_path",
                "anomaly_status",
                "risk_level",
                "risk_score",
                "recommended_action",
                "report_path",
                "processing_time_ms",
            ]

            with open(p, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                writer.writeheader()
                for rec in records:
                    writer.writerow(rec.to_dict())

            logger.info("[HistoryManager] Exported %d records to CSV: %s", len(records), p)
            return p
        except Exception as exc:
            raise ExportError(f"Failed to export CSV: {exc}") from exc

    def export_json(
        self,
        output_path: Union[str, Path] = "docs/sample_prediction_history.json",
        records: Optional[List[PredictionRecord]] = None,
    ) -> Path:
        """
        Export prediction records to JSON file.

        Parameters
        ----------
        output_path : str or Path
            Target JSON file path.
        records : List[PredictionRecord], optional
            Records to export. If None, exports all records in database.

        Returns
        -------
        Path
        """
        try:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)

            if records is None:
                records = self.get_history(limit=10_000)

            data = [rec.to_dict() for rec in records]
            p.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

            logger.info("[HistoryManager] Exported %d records to JSON: %s", len(records), p)
            return p
        except Exception as exc:
            raise ExportError(f"Failed to export JSON: {exc}") from exc

    # ------------------------------------------------------------------
    # Extraction Helpers
    # ------------------------------------------------------------------

    def _extract_pred(self, pred: Any) -> tuple:
        if isinstance(pred, dict):
            req_id = pred.get("prediction_id") or pred.get("request_id") or "REQ-DEMO-001"
            patient_id = pred.get("patient_id", "PT-DEMO-001")
            case_id = pred.get("case_id", "CASE-DEMO-001")
            modality = pred.get("modality", "chest_xray")
            disease = pred.get("predicted_disease") or pred.get("predicted_class") or "Unknown"
            conf = float(pred.get("confidence", 0.95))
            lat = float(pred.get("processing_time_ms", 12.5))
        else:
            req_id = getattr(pred, "prediction_id", None) or getattr(pred, "request_id", "REQ-DEMO-001")
            patient_id = getattr(pred, "patient_id", "PT-DEMO-001")
            case_id = getattr(pred, "case_id", "CASE-DEMO-001")
            modality = getattr(pred, "modality", "chest_xray")
            disease = getattr(pred, "predicted_disease", None) or getattr(pred, "predicted_class", "Unknown")
            conf = float(getattr(pred, "confidence", 0.95))
            lat = float(getattr(pred, "processing_time_ms", 12.5))

        return req_id, patient_id, case_id, modality, disease, conf, lat

    def _extract_exp(self, exp: Optional[Any]) -> tuple:
        if exp is None:
            return "Explainability not requested.", "", 0.0
        if isinstance(exp, dict):
            summary = exp.get("summary") or f"{exp.get('method', 'GradCAM')} saliency map generated."
            path = exp.get("heatmap_path") or exp.get("overlay_path") or ""
            lat = float(exp.get("processing_time_ms", 8.0))
        else:
            summary = getattr(exp, "summary", "GradCAM saliency map generated.")
            path = getattr(exp, "heatmap_path", None) or getattr(exp, "overlay_path", "")
            lat = float(getattr(exp, "processing_time_ms", 8.0))
        return summary, str(path), lat

    def _extract_anomaly(self, anomaly: Optional[Any]) -> tuple:
        if anomaly is None:
            return "KNOWN_DISEASE", "LOW", 0.1, "ROUTINE_FOLLOW_UP", 0.0
        if isinstance(anomaly, dict):
            status = anomaly.get("status", "KNOWN_DISEASE")
            risk = anomaly.get("risk_level", "LOW")
            score = float(anomaly.get("risk_score", 0.1))
            rec = anomaly.get("recommended_action", "ROUTINE_FOLLOW_UP")
            lat = float(anomaly.get("processing_time_ms", 4.0))
        else:
            status = getattr(anomaly, "status", "KNOWN_DISEASE")
            if hasattr(status, "value"):
                status = status.value
            risk = getattr(anomaly, "risk_level", "LOW")
            if hasattr(risk, "value"):
                risk = risk.value
            score = float(getattr(anomaly, "risk_score", 0.1))
            rec = getattr(anomaly, "recommended_action", "ROUTINE_FOLLOW_UP")
            if hasattr(rec, "value"):
                rec = rec.value
            lat = float(getattr(anomaly, "processing_time_ms", 4.0))
        return str(status), str(risk), score, str(rec), lat


__all__ = ["HistoryManager"]
