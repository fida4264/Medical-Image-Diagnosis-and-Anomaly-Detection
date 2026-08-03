"""
query_engine.py — Module 19: Prediction History & Database
Medical AI Platform — Flexible query builder and filtering engine for prediction history.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from medical_ai_engine.database.database_exceptions import QueryEngineError
from medical_ai_engine.database.database_manager import DatabaseManager
from medical_ai_engine.database.prediction_repository import PredictionRecord, PredictionRepository

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Rich query engine for searching, filtering, and paginating prediction history records.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager
        self._repo = PredictionRepository(db_manager)

    def get_by_id(self, prediction_id: str) -> Optional[PredictionRecord]:
        """Get prediction by ID."""
        return self._repo.get_by_id(prediction_id)

    def get_latest(self, limit: int = 10) -> List[PredictionRecord]:
        """Fetch latest N predictions ordered by created_at DESC."""
        return self.search(limit=limit, order_by="created_at", order_dir="DESC")

    def search(
        self,
        modality: Optional[str] = None,
        disease: Optional[str] = None,
        risk_level: Optional[str] = None,
        anomaly_status: Optional[str] = None,
        patient_id: Optional[str] = None,
        min_confidence: Optional[float] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
        order_by: str = "created_at",
        order_dir: str = "DESC",
    ) -> List[PredictionRecord]:
        """
        Execute filtered multi-parameter search over prediction history.

        Parameters
        ----------
        modality : str, optional
            Filter by image modality (e.g. 'chest_xray').
        disease : str, optional
            Filter by predicted disease.
        risk_level : str, optional
            Filter by risk level ('LOW', 'MODERATE', 'HIGH', 'CRITICAL').
        anomaly_status : str, optional
            Filter by anomaly status ('KNOWN_DISEASE', 'UNKNOWN_DISEASE', 'OUT_OF_DISTRIBUTION').
        patient_id : str, optional
            Filter by patient ID.
        min_confidence : float, optional
            Filter predictions with confidence >= min_confidence.
        start_date : str, optional
            ISO date filter start (inclusive).
        end_date : str, optional
            ISO date filter end (inclusive).
        limit : int
            Maximum records to return.
        offset : int
            Pagination offset.
        order_by : str
            Column to sort by ('created_at', 'confidence', etc.).
        order_dir : str
            Sort direction ('ASC' or 'DESC').

        Returns
        -------
        List[PredictionRecord]
        """
        valid_cols = {"created_at", "confidence", "modality", "predicted_disease", "risk_level", "processing_time_ms"}
        if order_by not in valid_cols:
            order_by = "created_at"

        order_dir_clean = "DESC" if order_dir.upper() == "DESC" else "ASC"

        where_clauses: List[str] = []
        params: List[Any] = []

        if modality:
            where_clauses.append("modality = ?")
            params.append(modality)
        if disease:
            where_clauses.append("predicted_disease LIKE ?")
            params.append(f"%{disease}%")
        if risk_level:
            where_clauses.append("risk_level = ?")
            params.append(risk_level.upper())
        if anomaly_status:
            where_clauses.append("anomaly_status = ?")
            params.append(anomaly_status.upper())
        if patient_id:
            where_clauses.append("patient_id = ?")
            params.append(patient_id)
        if min_confidence is not None:
            where_clauses.append("confidence >= ?")
            params.append(min_confidence)
        if start_date:
            where_clauses.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            where_clauses.append("created_at <= ?")
            params.append(end_date)

        where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
        sql = f"""
        SELECT * FROM predictions
        {where_sql}
        ORDER BY {order_by} {order_dir_clean}
        LIMIT ? OFFSET ?;
        """
        params.extend([limit, offset])

        try:
            with self.db_manager.connection_scope() as conn:
                cursor = conn.execute(sql, params)
                rows = cursor.fetchall()
                return [self._repo._row_to_record(row) for row in rows]
        except Exception as exc:
            raise QueryEngineError(f"Search query failed: {exc}") from exc


__all__ = ["QueryEngine"]
