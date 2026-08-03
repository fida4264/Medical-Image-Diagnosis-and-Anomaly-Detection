"""
Medical AI Engine — Routing Exception Hierarchy

Defines exception types for routing operations.
Inherits from EngineExecutionException (ENG-300 block).
"""

from __future__ import annotations

from typing import Optional
from medical_ai_engine.core.exceptions import EngineExecutionException


class RoutingException(EngineExecutionException):
    """Base exception for routing stage failures."""

    ERROR_CODE = "ENG-310"

    def __init__(
        self,
        message: str = "Routing operation failed.",
        modality: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.modality = modality
        if modality:
            message = f"{message} [modality={modality}]"
        super().__init__(message, error_code=self.ERROR_CODE, **kwargs)


class ModalityClassificationException(RoutingException):
    """Raised when Module 6 ModalityPredictor fails during inference."""

    ERROR_CODE = "ENG-311"

    def __init__(
        self,
        message: str = "Modality classification failed.",
        cause: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        super().__init__(message, cause=cause, **kwargs)
        self.error_code = self.ERROR_CODE


class RoutingConfigurationException(RoutingException):
    """Raised when routing configuration (routing.yaml) is invalid or missing."""

    ERROR_CODE = "ENG-312"

    def __init__(
        self,
        message: str = "Routing configuration error.",
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.config_path = config_path
        if config_path:
            message = f"{message} [config_path={config_path}]"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE
