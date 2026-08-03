"""
prediction_repository.py — Module 19: Prediction History & Database
Medical AI Platform — Data Access Object (DAO) for prediction database records.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from medical_ai_engine.database.database_exceptions import RepositoryError
from medical_ai_engine.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PredictionRecord:
    """
    Database record representing a stored prediction and associated diagnostic artifacts.
    """

    prediction_id: str
    created_at: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    patient_id: str = "PT-DEMO-001"
    case_id: str = "CASE-DEMO-001"
    modality: str = "chest_xray"
    predicted_disease: str = "Pneumonia"
    confidence: float = 0.95
    explainability_summary: str = ""
    heatmap_path: str = ""
    anomaly_status: str = "KNOWN_DISEASE"
    risk_level: str = "LOW"
    risk_score: float = 0.1
    recommended_action: str = "ROUTINE_FOLLOW_UP"
    report_path: str = ""
    processing_time_ms: float = 24.5
    extra_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["confidence_pct"] = round(self.confidence * 100.0 if self.confidence <= 1.0 else self.confidence, 2)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class PredictionRepository:
    """
    CRUD Data Access Object for `predictions` SQLite table.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def save(self, record: PredictionRecord) -> str:
        """
        Insert or replace a prediction record.

        Returns prediction_id.
        """
        sql = """
        INSERT OR REPLACE INTO predictions (
            prediction_id, created_at, patient_id, case_id, modality,
            predicted_disease, confidence, explainability_summary, heatmap_path,
            anomaly_status, risk_level, risk_score, recommended_action,
            report_path, processing_time_ms, extra_metadata
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            meta_json = json.dumps(record.extra_metadata, default=str)
            params = (
                record.prediction_id,
                record.created_at,
                record.patient_id,
                record.case_id,
                record.modality,
                record.predicted_disease,
                record.confidence,
                record.explainability_summary,
                record.heatmap_path,
                record.anomaly_status,
                record.risk_level,
                record.risk_score,
                record.recommended_action,
                record.report_path,
                record.processing_time_ms,
                meta_json,
            )
            with self.db_manager.connection_scope() as conn:
                conn.execute(sql, params)

            logger.info("[PredictionRepository] Saved prediction record %s", record.prediction_id)
            return record.prediction_id
        except Exception as exc:
            raise RepositoryError(f"Failed to save prediction record {record.prediction_id}: {exc}") from exc

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Fetch a single prediction record by ID."""
        sql = "SELECT * FROM predictions WHERE prediction_id = ?;"
        try:
            with self.db_manager.connection_scope() as conn:
                cursor = conn.execute(sql, (prediction_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return self._row_to_record(row)
        except Exception as exc:
            raise RepositoryError(f"Failed to fetch prediction record {prediction_id}: {exc}") from exc

    def update(self, record: PredictionRecord) -> bool:
        """Update an existing prediction record."""
        return self.save(record) == record.prediction_id

    def delete(self, prediction_id: str) -> bool:
        """Delete prediction record by ID."""
        sql = "DELETE FROM predictions WHERE prediction_id = ?;"
        try:
            with self.db_manager.connection_scope() as conn:
                cursor = conn.execute(sql, (prediction_id,))
                return cursor.rowcount > 0
        except Exception as exc:
            raise RepositoryError(f"Failed to delete prediction record {prediction_id}: {exc}") from exc

    def count(self) -> int:
        """Get total prediction records count."""
        sql = "SELECT COUNT(*) FROM predictions;"
        try:
            with self.db_manager.connection_scope() as conn:
                cursor = conn.execute(sql)
                return cursor.fetchone()[0]
        except Exception as exc:
            raise RepositoryError(f"Failed to count predictions: {exc}") from exc

    def _row_to_record(self, row: Any) -> PredictionRecord:
        """Convert sqlite3.Row to PredictionRecord."""
        d = dict(row)
        meta_str = d.get("extra_metadata") or "{}"
        try:
            extra_meta = json.loads(meta_str)
        except Exception:
            extra_meta = {}

        return PredictionRecord(
            prediction_id=d["prediction_id"],
            created_at=d["created_at"],
            patient_id=d.get("patient_id") or "PT-DEMO-001",
            case_id=d.get("case_id") or "CASE-DEMO-001",
            modality=d["modality"],
            predicted_disease=d["predicted_disease"],
            confidence=float(d["confidence"]),
            explainability_summary=d.get("explainability_summary") or "",
            heatmap_path=d.get("heatmap_path") or "",
            anomaly_status=d.get("anomaly_status") or "KNOWN_DISEASE",
            risk_level=d.get("risk_level") or "LOW",
            risk_score=float(d.get("risk_score") or 0.0),
            recommended_action=d.get("recommended_action") or "ROUTINE_FOLLOW_UP",
            report_path=d.get("report_path") or "",
            processing_time_ms=float(d.get("processing_time_ms") or 0.0),
            extra_metadata=extra_meta,
        )


__all__ = ["PredictionRecord", "PredictionRepository"]
