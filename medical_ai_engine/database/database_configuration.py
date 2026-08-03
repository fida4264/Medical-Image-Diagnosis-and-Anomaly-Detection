"""
database_configuration.py — Module 19: Prediction History & Database
Medical AI Platform — Configuration dataclass for SQLite database management.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from medical_ai_engine.database.database_exceptions import DatabaseConfigurationError


@dataclass
class DatabaseConfiguration:
    """
    Configuration options for the prediction history database subsystem.

    Attributes
    ----------
    database_path : str
        SQLite database file path (e.g. "medical_ai.db" or ":memory:").
    auto_create_tables : bool
        Whether to automatically create schema tables on initialization.
    enable_wal_mode : bool
        Whether to enable Write-Ahead Logging (WAL) for concurrent performance.
    connection_timeout_seconds : float
        SQLite connection timeout.
    export_directory : str
        Directory to save exported CSV/JSON files.
    """

    database_path: str = "medical_ai.db"
    auto_create_tables: bool = True
    enable_wal_mode: bool = True
    connection_timeout_seconds: float = 30.0
    export_directory: str = "docs"

    def __post_init__(self) -> None:
        if not self.database_path.strip():
            raise DatabaseConfigurationError("database_path cannot be empty.")
        if self.connection_timeout_seconds <= 0:
            raise DatabaseConfigurationError("connection_timeout_seconds must be > 0.")

    @classmethod
    def default(cls) -> DatabaseConfiguration:
        """Production default configuration."""
        return cls()

    @classmethod
    def fast_test(cls) -> DatabaseConfiguration:
        """Fast configuration for unit testing using in-memory database."""
        return cls(
            database_path=":memory:",
            enable_wal_mode=False,
            export_directory="docs",
        )


__all__ = ["DatabaseConfiguration"]
