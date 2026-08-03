"""
__init__.py — Module 19: Prediction History & Database
Medical AI Platform — Package public API for database subsystem.
"""

from __future__ import annotations

from medical_ai_engine.database.database_configuration import DatabaseConfiguration
from medical_ai_engine.database.database_exceptions import (
    DatabaseConfigurationError,
    DatabaseConnectionError,
    DatabaseException,
    ExportError,
    QueryEngineError,
    RepositoryError,
    StatisticsEngineError,
)
from medical_ai_engine.database.database_manager import DatabaseManager
from medical_ai_engine.database.history_manager import HistoryManager
from medical_ai_engine.database.prediction_repository import PredictionRecord, PredictionRepository
from medical_ai_engine.database.query_engine import QueryEngine
from medical_ai_engine.database.statistics_engine import PlatformDatabaseStatistics, StatisticsEngine

__all__ = [
    # Main Facade & Managers
    "HistoryManager",
    "DatabaseManager",
    "PredictionRepository",
    "QueryEngine",
    "StatisticsEngine",
    # Data Models
    "PredictionRecord",
    "PlatformDatabaseStatistics",
    # Configuration
    "DatabaseConfiguration",
    # Exceptions
    "DatabaseException",
    "DatabaseConnectionError",
    "RepositoryError",
    "QueryEngineError",
    "StatisticsEngineError",
    "ExportError",
    "DatabaseConfigurationError",
]
