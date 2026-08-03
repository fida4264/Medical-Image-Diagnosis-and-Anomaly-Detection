"""
statistics_engine.py — Module 19: Prediction History & Database
Medical AI Platform — Analytics and aggregation statistics engine.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Dict

from medical_ai_engine.database.database_exceptions import StatisticsEngineError
from medical_ai_engine.database.database_manager import DatabaseManager

logger = logging.getLogger(__name__)


@dataclass
class PlatformDatabaseStatistics:
    """
    Summary analytics container for database prediction history.
    """

    total_predictions: int
    predictions_per_modality: Dict[str, int] = field(default_factory=dict)
    disease_frequency: Dict[str, int] = field(default_factory=dict)
    average_confidence: float = 0.0
    average_processing_time_ms: float = 0.0
    risk_level_distribution: Dict[str, int] = field(default_factory=dict)
    anomaly_status_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["average_confidence"] = round(self.average_confidence, 4)
        d["average_processing_time_ms"] = round(self.average_processing_time_ms, 2)
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)


class StatisticsEngine:
    """
    Computes analytical aggregations over stored prediction history.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        self.db_manager = db_manager

    def compute_statistics(self) -> PlatformDatabaseStatistics:
        """
        Compute platform-wide database analytics.

        Returns
        -------
        PlatformDatabaseStatistics
        """
        try:
            with self.db_manager.connection_scope() as conn:
                # 1. Total predictions & averages
                row1 = conn.execute(
                    "SELECT COUNT(*), AVG(confidence), AVG(processing_time_ms) FROM predictions;"
                ).fetchone()

                total = row1[0] or 0
                avg_conf = float(row1[1] or 0.0)
                avg_lat = float(row1[2] or 0.0)

                if total == 0:
                    return PlatformDatabaseStatistics(total_predictions=0)

                # 2. Modality breakdown
                mod_rows = conn.execute(
                    "SELECT modality, COUNT(*) FROM predictions GROUP BY modality ORDER BY COUNT(*) DESC;"
                ).fetchall()
                per_modality = {row[0]: row[1] for row in mod_rows}

                # 3. Disease frequency
                disease_rows = conn.execute(
                    "SELECT predicted_disease, COUNT(*) FROM predictions GROUP BY predicted_disease ORDER BY COUNT(*) DESC;"
                ).fetchall()
                disease_freq = {row[0]: row[1] for row in disease_rows}

                # 4. Risk level distribution
                risk_rows = conn.execute(
                    "SELECT risk_level, COUNT(*) FROM predictions GROUP BY risk_level;"
                ).fetchall()
                risk_dist = {row[0]: row[1] for row in risk_rows}

                # 5. Anomaly status distribution
                anom_rows = conn.execute(
                    "SELECT anomaly_status, COUNT(*) FROM predictions GROUP BY anomaly_status;"
                ).fetchall()
                anom_dist = {row[0]: row[1] for row in anom_rows}

            return PlatformDatabaseStatistics(
                total_predictions=total,
                predictions_per_modality=per_modality,
                disease_frequency=disease_freq,
                average_confidence=avg_conf,
                average_processing_time_ms=avg_lat,
                risk_level_distribution=risk_dist,
                anomaly_status_distribution=anom_dist,
            )
        except Exception as exc:
            raise StatisticsEngineError(f"Failed to compute database statistics: {exc}") from exc


__all__ = ["PlatformDatabaseStatistics", "StatisticsEngine"]
