"""
api/routes/system.py — Module 21: Production REST API
System health check, liveness probe, readiness probe, and platform telemetry endpoints.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, Optional
from fastapi import APIRouter, Depends, Response, status

from api import __version__
from api.dependencies import get_db_manager, get_engine, verify_api_key
from api.schemas import HealthStatusResponse, ReadinessStatusResponse, SystemInfoResponse
from medical_ai_engine import MedicalAIEngine
from medical_ai_engine.database.database_manager import DatabaseManager

router = APIRouter(prefix="", tags=["System & Telemetry"])

_START_TIME = time.time()

_KNOWN_ADAPTERS = [
    "brain_mri", "chest_xray", "brain_ct",
    "histopathology", "retina", "skin", "ultrasound",
]


@router.get(
    "/",
    status_code=status.HTTP_200_OK,
    summary="API Root Endpoint",
    description="Returns platform identification, version, status, and navigation links.",
)
async def get_root() -> Dict[str, Any]:
    from api.settings import settings
    return {
        "title": "Medical Image Diagnosis & Anomaly Detection REST API",
        "version": __version__,
        "status": "ONLINE",
        "docs_url": "/docs" if settings.ENABLE_DOCS else "disabled",
        "redoc_url": "/redoc" if settings.ENABLE_DOCS else "disabled",
        "openapi_url": "/openapi.json" if settings.ENABLE_DOCS else "disabled",
        "health_url": "/health",
    }


@router.get(
    "/health/liveness",
    status_code=status.HTTP_200_OK,
    summary="Kubernetes / Cloud Liveness Probe",
    description="Lightweight liveness ping returning 200 OK if server process is running.",
)
async def get_liveness() -> Dict[str, str]:
    return {"status": "UP"}


@router.get(
    "/health/readiness",
    response_model=ReadinessStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Kubernetes / Cloud Readiness Probe",
    description="Readiness check verifying engine initialization and database connection scope.",
)
async def get_readiness(
    response: Response,
    engine: MedicalAIEngine = Depends(get_engine),
    db: DatabaseManager = Depends(get_db_manager),
) -> ReadinessStatusResponse:
    # Check engine
    engine_ready = False
    try:
        h = engine.get_health_status()
        engine_ready = h.get("current_state") == "INITIALIZED" or h.get("status") == "HEALTHY"
    except Exception:
        engine_ready = False

    # Check database
    db_ready = False
    try:
        with db.connection_scope() as conn:
            conn.execute("SELECT 1;")
        db_ready = True
    except Exception:
        db_ready = False

    is_ready = engine_ready and db_ready
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessStatusResponse(
        status="READY" if is_ready else "NOT_READY",
        engine_ready=engine_ready,
        database_ready=db_ready,
    )


@router.get(
    "/health",
    response_model=HealthStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Detailed System Health Telemetry",
    description="Returns real-time health status of REST API, MedicalAIEngine state, SQLite database connectivity, loaded AI adapters, and uptime.",
)
async def get_health(
    engine: MedicalAIEngine = Depends(get_engine),
    db: DatabaseManager = Depends(get_db_manager),
    _auth: Optional[str] = Depends(verify_api_key),
) -> HealthStatusResponse:
    try:
        health_info = engine.get_health_status()
        engine_state = str(health_info.get("current_state", "INITIALIZED"))
    except Exception:
        engine_state = "ERROR"

    try:
        with db.connection_scope() as conn:
            conn.execute("SELECT 1;")
        db_status = "OK"
    except Exception:
        db_status = "ERROR"

    overall_status = "HEALTHY" if (engine_state not in ("ERROR", "FAILED") and db_status == "OK") else "DEGRADED"
    uptime_sec = round(time.time() - _START_TIME, 2)

    return HealthStatusResponse(
        status=overall_status,
        api_version=__version__,
        engine_state=engine_state,
        database_status=db_status,
        loaded_adapters=_KNOWN_ADAPTERS[:],
        uptime_seconds=uptime_sec,
    )


@router.get(
    "/system/info",
    response_model=SystemInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Platform Architecture & Technology Metadata",
    description="Retrieve comprehensive metadata regarding platform capabilities, supported imaging modalities, pipeline stages, and technology stack.",
)
async def get_system_info() -> SystemInfoResponse:
    return SystemInfoResponse(
        platform_name="Medical Image Diagnosis & Anomaly Detection Platform",
        platform_version=__version__,
        supported_modalities=_KNOWN_ADAPTERS[:],
        pipeline_stages=[
            "Stage 1: InputValidationStage",
            "Stage 2: ImagePreprocessingStage",
            "Stage 3: ModalityClassificationStage",
            "Stage 4: RoutingStage",
            "Stage 5: AdapterPredictionStage",
            "Stage 6: ConfidenceAssessmentStage",
            "Stage 7: ClinicalReportHookStage",
            "Stage 8: ResponsePreparationStage",
        ],
        database_backend="SQLite 3 (Module 19 WAL Mode)",
        xai_framework="GradCAM / LayerCAM Saliency Engine (Module 15)",
        anomaly_engine="Mahalanobis OOD & Autoencoder Fusion (Module 16)",
        tech_stack=[
            "Python 3.12",
            "FastAPI 0.141",
            "PyTorch 2.0",
            "Pydantic V2",
            "SQLite 3",
            "PIL / Pillow",
            "Gradio 6.0",
        ],
    )
