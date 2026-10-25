"""
api/main.py — Module 21: Production REST API Entry Point
Medical Image Diagnosis & Anomaly Detection Platform.
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from api import __version__
from api.dependencies import get_db_manager, get_engine
from api.exceptions import register_exception_handlers
from api.middleware import setup_middleware
from api.routes import benchmark, history, prediction, reports, system
from api.settings import settings
from pipeline_v2.api_router import router as v2_pipeline_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalAI.API")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager. Pre-warms engine singleton and
    verifies database schema on API startup.
    """
    logger.info(f"Initializing Medical AI REST API (v{__version__}) [Env: {settings.ENVIRONMENT}]...")

    # Initialize MedicalAIEngine singleton
    try:
        engine = get_engine()
        logger.info(f"MedicalAIEngine initialized: state={engine.state.name}")
    except Exception as exc:
        logger.error(f"Failed to initialize MedicalAIEngine on startup: {exc}")

    # Initialize DatabaseManager singleton
    try:
        db = get_db_manager()
        logger.info(f"DatabaseManager initialized: file={db.config.database_path}")
    except Exception as exc:
        logger.error(f"Failed to initialize DatabaseManager on startup: {exc}")

    yield

    logger.info("Shutting down Medical AI REST API services...")


def create_app() -> FastAPI:
    """Construct and configure the production FastAPI application."""
    # Control Swagger UI / ReDoc availability based on settings.ENABLE_DOCS
    docs_url = "/docs" if settings.ENABLE_DOCS else None
    redoc_url = "/redoc" if settings.ENABLE_DOCS else None
    openapi_url = "/openapi.json" if settings.ENABLE_DOCS else None

    app = FastAPI(
        title="Medical Image Diagnosis & Anomaly Detection REST API",
        description=(
            "Production-grade, high-performance REST API for automated multi-modality "
            "medical image classification, GradCAM explainability, out-of-distribution "
            "anomaly detection, report generation, and prediction history auditing."
        ),
        version=__version__,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
        lifespan=lifespan,
    )

    # Register custom exception handlers
    register_exception_handlers(app)

    # Configure middleware (CORS, security headers, rate limiter, timing, tracing)
    setup_middleware(app)

    # Register routers
    app.include_router(system.router)
    app.include_router(prediction.router)
    app.include_router(history.router)
    app.include_router(reports.router)
    app.include_router(benchmark.router)
    app.include_router(v2_pipeline_router)

    # When documentation is disabled, return structured JSON instead of broken page
    if not settings.ENABLE_DOCS:
        @app.get("/docs", include_in_schema=False)
        @app.get("/redoc", include_in_schema=False)
        @app.get("/openapi.json", include_in_schema=False)
        async def disabled_docs_handler(request: Request):
            return JSONResponse(
                status_code=status.HTTP_404_NOT_FOUND,
                content={
                    "status": "disabled",
                    "message": "API documentation is disabled in this environment.",
                    "code": "DOCS_DISABLED",
                    "path": request.url.path,
                },
            )

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Uvicorn production server on http://127.0.0.1:8000 ...")
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
