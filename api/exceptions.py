"""
api/exceptions.py — Module 21: Production REST API
Sanitized HTTP exceptions and global FastAPI exception handlers.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.schemas import ErrorDetail, ErrorResponse
from api.settings import settings

logger = logging.getLogger("MedicalAI.API.Exceptions")


class APIException(Exception):
    """Base class for all Medical AI REST API exceptions."""

    def __init__(
        self,
        message: str,
        error_code: str = "API_ERROR",
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        details: Any = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details


class InvalidImageException(APIException):
    def __init__(self, message: str = "Uploaded file is not a valid medical image.") -> None:
        super().__init__(
            message=message,
            error_code="INVALID_IMAGE",
            status_code=status.HTTP_400_BAD_REQUEST,
        )


class FileTooLargeException(APIException):
    def __init__(self, max_mb: float = 20.0) -> None:
        super().__init__(
            message=f"Uploaded file exceeds maximum allowed size of {max_mb:.1f} MB.",
            error_code="FILE_TOO_LARGE",
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        )


class UnsupportedModalityException(APIException):
    def __init__(self, modality: str) -> None:
        super().__init__(
            message=f"Modality '{modality}' is unsupported.",
            error_code="UNSUPPORTED_MODALITY",
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        )


class PredictionFailedException(APIException):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Prediction pipeline execution failed: {reason}",
            error_code="PREDICTION_FAILED",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class PredictionNotFoundException(APIException):
    def __init__(self, prediction_id: str) -> None:
        super().__init__(
            message=f"Prediction record '{prediction_id}' not found.",
            error_code="PREDICTION_NOT_FOUND",
            status_code=status.HTTP_404_NOT_FOUND,
        )


class ReportExportException(APIException):
    def __init__(self, reason: str) -> None:
        super().__init__(
            message=f"Report generation/export failed: {reason}",
            error_code="REPORT_EXPORT_ERROR",
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


class UnauthorizedAPIKeyException(APIException):
    def __init__(self, message: str = "Invalid or missing API key.") -> None:
        super().__init__(
            message=message,
            error_code="UNAUTHORIZED",
            status_code=status.HTTP_401_UNAUTHORIZED,
        )


class RateLimitExceededException(APIException):
    def __init__(self, limit: int, retry_after_sec: int = 60) -> None:
        super().__init__(
            message=f"Rate limit exceeded ({limit} requests/minute). Please retry in {retry_after_sec} seconds.",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register custom exception handlers on the FastAPI app instance."""

    @app.exception_handler(APIException)
    async def custom_api_exception_handler(request: Request, exc: APIException) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "N/A")
        logger.warning(f"APIException [{exc.error_code}] ReqID={req_id} Path={request.url.path}: {exc.message}")

        headers = {}
        if exc.status_code == status.HTTP_429_TOO_MANY_REQUESTS:
            headers["Retry-After"] = "60"

        payload = ErrorResponse(
            error_code=exc.error_code,
            message=exc.message,
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
        )
        return JSONResponse(status_code=exc.status_code, content=payload.model_dump(), headers=headers)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "N/A")
        logger.warning(f"RequestValidationError ReqID={req_id} Path={request.url.path}: {exc.errors()}")
        details = [
            ErrorDetail(
                loc=[str(l) for l in err.get("loc", [])],
                msg=str(err.get("msg", "")),
                type=str(err.get("type", "")),
            )
            for err in exc.errors()
        ]
        payload = ErrorResponse(
            error_code="VALIDATION_ERROR",
            message="Request input validation failed.",
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
            details=details,
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content=payload.model_dump(),
        )

    @app.exception_handler(Exception)
    async def global_unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        req_id = getattr(request.state, "request_id", "N/A")
        logger.error(f"Unhandled exception ReqID={req_id} Path={request.url.path}: {exc}", exc_info=True)

        # In production mode, hide internal raw stack trace error details
        msg = "An unexpected internal server error occurred." if settings.is_production else str(exc)

        payload = ErrorResponse(
            error_code="INTERNAL_SERVER_ERROR",
            message=msg,
            request_id=req_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            path=request.url.path,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=payload.model_dump(),
        )
