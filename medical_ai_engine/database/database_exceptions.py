"""
database_exceptions.py — Module 19: Prediction History & Database
Medical AI Platform — Typed exception hierarchy for database subsystem.
"""

from __future__ import annotations


class DatabaseException(Exception):
    """Base exception for all database subsystem errors."""


class DatabaseConnectionError(DatabaseException):
    """Raised when connecting or creating the SQLite database fails."""


class RepositoryError(DatabaseException):
    """Raised when CRUD operations fail."""


class QueryEngineError(DatabaseException):
    """Raised when building or executing database queries fails."""


class StatisticsEngineError(DatabaseException):
    """Raised when computing database statistics fails."""


class ExportError(DatabaseException):
    """Raised when exporting prediction history to CSV or JSON fails."""


class DatabaseConfigurationError(DatabaseException):
    """Raised when database configuration options are invalid."""


__all__ = [
    "DatabaseException",
    "DatabaseConnectionError",
    "RepositoryError",
    "QueryEngineError",
    "StatisticsEngineError",
    "ExportError",
    "DatabaseConfigurationError",
]
