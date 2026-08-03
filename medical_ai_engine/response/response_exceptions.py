"""
Medical AI Engine — Response Exception Hierarchy

Defines exceptions raised during response building, validation, and serialization.
"""

from __future__ import annotations

from typing import Optional, List
from medical_ai_engine.core.exceptions import EngineExecutionException


class ResponseException(EngineExecutionException):
    """Base exception for all response subsystem failures."""

    ERROR_CODE = "ENG-330"

    def __init__(
        self,
        message: str = "Response operation failed.",
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE


class ResponseBuildException(ResponseException):
    """Raised when ResponseBuilder cannot construct a PredictionResult."""

    ERROR_CODE = "ENG-331"

    def __init__(
        self,
        message: str = "Failed to build PredictionResult.",
        missing_fields: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.missing_fields = missing_fields or []
        if self.missing_fields:
            message = f"{message} Missing fields: {self.missing_fields}"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE


class ResponseValidationException(ResponseException):
    """Raised when PredictionResult fails schema or consistency validation."""

    ERROR_CODE = "ENG-332"

    def __init__(
        self,
        message: str = "PredictionResult validation failed.",
        validation_errors: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.validation_errors = validation_errors or []
        if self.validation_errors:
            details = "; ".join(self.validation_errors)
            message = f"{message} Errors: [{details}]"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE


class ResponseSerializationException(ResponseException):
    """Raised when PredictionResult serialization to JSON or dict fails."""

    ERROR_CODE = "ENG-333"

    def __init__(
        self,
        message: str = "PredictionResult serialization failed.",
        format_requested: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.format_requested = format_requested
        if format_requested:
            message = f"{message} [format={format_requested}]"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE
