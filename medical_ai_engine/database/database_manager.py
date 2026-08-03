"""
database_manager.py — Module 19: Prediction History & Database
Medical AI Platform — SQLite connection management, schema initialization, and migrations.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Optional

from medical_ai_engine.database.database_configuration import DatabaseConfiguration
from medical_ai_engine.database.database_exceptions import DatabaseConnectionError

logger = logging.getLogger(__name__)

CREATE_PREDICTIONS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    prediction_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    patient_id TEXT,
    case_id TEXT,
    modality TEXT NOT NULL,
    predicted_disease TEXT NOT NULL,
    confidence REAL NOT NULL,
    explainability_summary TEXT,
    heatmap_path TEXT,
    anomaly_status TEXT,
    risk_level TEXT,
    risk_score REAL,
    recommended_action TEXT,
    report_path TEXT,
    processing_time_ms REAL,
    extra_metadata TEXT
);
"""

CREATE_INDEXES_SQL = [
    "CREATE INDEX IF NOT EXISTS idx_modality ON predictions(modality);",
    "CREATE INDEX IF NOT EXISTS idx_disease ON predictions(predicted_disease);",
    "CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_risk_level ON predictions(risk_level);",
    "CREATE INDEX IF NOT EXISTS idx_patient_id ON predictions(patient_id);",
]


class DatabaseManager:
    """
    Manages SQLite database connections, schema creation, and migrations.

    Thread-safe connection handling using thread-local storage or context managers.
    """

    def __init__(self, config: Optional[DatabaseConfiguration] = None) -> None:
        self.config = config or DatabaseConfiguration.default()
        self._local = threading.local()
        self._lock = threading.Lock()

        if self.config.database_path != ":memory:":
            db_file = Path(self.config.database_path)
            db_file.parent.mkdir(parents=True, exist_ok=True)

        if self.config.auto_create_tables:
            self.initialize_schema()

    def get_connection(self) -> sqlite3.Connection:
        """Get or create thread-local SQLite connection."""
        conn = getattr(self._local, "connection", None)
        if conn is None:
            try:
                conn = sqlite3.connect(
                    self.config.database_path,
                    timeout=self.config.connection_timeout_seconds,
                    check_same_thread=False,
                )
                conn.row_factory = sqlite3.Row
                if self.config.enable_wal_mode and self.config.database_path != ":memory:":
                    conn.execute("PRAGMA journal_mode=WAL;")
                self._local.connection = conn
            except Exception as exc:
                raise DatabaseConnectionError(f"Failed to connect to SQLite: {exc}") from exc
        return conn

    @contextmanager
    def connection_scope(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager yielding a transactional connection scope."""
        conn = self.get_connection()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise

    def initialize_schema(self) -> None:
        """Create schema tables and secondary indexes."""
        try:
            with self.connection_scope() as conn:
                conn.execute(CREATE_PREDICTIONS_TABLE_SQL)
                for idx_sql in CREATE_INDEXES_SQL:
                    conn.execute(idx_sql)
            logger.info("[DatabaseManager] Database schema initialized: %s", self.config.database_path)
        except Exception as exc:
            raise DatabaseConnectionError(f"Schema initialization failed: {exc}") from exc

    def close(self) -> None:
        """Close thread-local connection."""
        conn = getattr(self._local, "connection", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.connection = None


__all__ = ["DatabaseManager"]
