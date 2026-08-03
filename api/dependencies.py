"""
api/dependencies.py — Module 21: Production REST API
FastAPI dependency injection providers, constant-time multi-key authentication, and role authorization.
"""

from __future__ import annotations

import secrets
import logging
from typing import Callable, List, Optional
from fastapi import Header, Query, Depends

from api.exceptions import UnauthorizedAPIKeyException
from api.settings import settings
from medical_ai_engine import MedicalAIEngine
from medical_ai_engine.database.database_manager import DatabaseManager
from medical_ai_engine.database.prediction_repository import PredictionRepository
from medical_ai_engine.database.history_manager import HistoryManager
from medical_ai_engine.database.query_engine import QueryEngine
from medical_ai_engine.database.statistics_engine import StatisticsEngine
from medical_ai_engine.reporting.report_generator import ReportGenerator
from medical_ai_engine.reporting.report_exporter import ReportExporter
from medical_ai_engine.benchmark.benchmark_runner import BenchmarkRunner

logger = logging.getLogger("MedicalAI.API.Dependencies")

# ── Singleton Instances ────────────────────────────────────────────────────────

_engine_instance: Optional[MedicalAIEngine] = None
_db_manager_instance: Optional[DatabaseManager] = None


def get_engine() -> MedicalAIEngine:
    """Return pre-warmed MedicalAIEngine singleton instance."""
    global _engine_instance
    if _engine_instance is None:
        logger.info("Initializing MedicalAIEngine singleton for REST API...")
        _engine_instance = MedicalAIEngine()
        _engine_instance.initialize()
    return _engine_instance


def get_db_manager() -> DatabaseManager:
    """Return DatabaseManager singleton instance (schema auto-initialized in __init__)."""
    global _db_manager_instance
    if _db_manager_instance is None:
        _db_manager_instance = DatabaseManager()
    return _db_manager_instance


def get_prediction_repository(
    db: DatabaseManager = Depends(get_db_manager)
) -> PredictionRepository:
    return PredictionRepository(db)


def get_query_engine(
    db: DatabaseManager = Depends(get_db_manager)
) -> QueryEngine:
    return QueryEngine(db)


def get_history_manager(
    db: DatabaseManager = Depends(get_db_manager)
) -> HistoryManager:
    return HistoryManager(db)


def get_statistics_engine(
    db: DatabaseManager = Depends(get_db_manager)
) -> StatisticsEngine:
    return StatisticsEngine(db)


def get_report_generator() -> ReportGenerator:
    return ReportGenerator()


def get_report_exporter() -> ReportExporter:
    return ReportExporter()


def get_benchmark_runner() -> BenchmarkRunner:
    return BenchmarkRunner()


# ── Authentication & Security Dependencies ────────────────────────────────────

def verify_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    api_key: Optional[str] = Query(default=None),
) -> Optional[str]:
    """
    Verify provided API key against configured valid keys using constant-time comparison
    (`secrets.compare_digest`) to mitigate timing side-channel attacks.

    Supports multiple API keys and key rotation.
    If AUTH_REQUIRED is False and no valid keys are configured, auth is bypassed.
    """
    valid_keys = settings.valid_api_keys

    # Bypassed if auth not required AND no valid keys set in settings
    if not settings.AUTH_REQUIRED and not valid_keys:
        return None

    provided_key = x_api_key or api_key
    if not provided_key:
        raise UnauthorizedAPIKeyException("API key header 'X-API-Key' or query parameter 'api_key' is required.")

    # Constant-time matching against all valid keys
    authenticated = False
    for valid_key in valid_keys:
        if secrets.compare_digest(provided_key, valid_key):
            authenticated = True
            break

    if not authenticated:
        logger.warning(f"Unauthorized API key attempt: prefix='{provided_key[:4]}...'")
        raise UnauthorizedAPIKeyException("Invalid or revoked API key.")

    return provided_key


def require_role(allowed_roles: List[str]) -> Callable:
    """
    Role-based access control (RBAC) dependency hook for future permission architecture.
    """
    def role_checker(key: Optional[str] = Depends(verify_api_key)) -> str:
        # Default implementation maps any valid key to 'clinician' role
        # Can be expanded with key-to-role lookup table in database or JWT tokens
        user_role = "clinician"
        if "admin" in key if key else False:
            user_role = "admin"

        if user_role not in allowed_roles and "admin" not in allowed_roles:
            raise UnauthorizedAPIKeyException(f"Role '{user_role}' insufficient. Required: {allowed_roles}")
        return user_role

    return role_checker
