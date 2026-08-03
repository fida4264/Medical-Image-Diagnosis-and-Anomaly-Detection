"""
Medical AI Engine — Integration Exceptions

Exception hierarchy for Module 14.2.6: End-to-End System Integration.
"""

from __future__ import annotations

from typing import Optional, List
from medical_ai_engine.core.exceptions import EngineExecutionException


class IntegrationException(EngineExecutionException):
    """Base exception for all system integration errors."""

    ERROR_CODE = "ENG-340"

    def __init__(
        self,
        message: str = "System integration failure.",
        **kwargs,
    ) -> None:
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE


class WorkflowExecutionException(IntegrationException):
    """Raised when an end-to-end prediction workflow fails during execution."""

    ERROR_CODE = "ENG-341"

    def __init__(
        self,
        message: str = "Workflow execution failed.",
        modality: Optional[str] = None,
        request_id: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.modality = modality
        if modality:
            message = f"{message} [modality={modality}]"
        super().__init__(message, request_id=request_id, **kwargs)
        self.error_code = self.ERROR_CODE


class IntegrationValidationException(IntegrationException):
    """Raised when end-to-end system integration validation fails."""

    ERROR_CODE = "ENG-342"

    def __init__(
        self,
        message: str = "Integration validation failed.",
        validation_errors: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.validation_errors = validation_errors or []
        if self.validation_errors:
            details = "; ".join(self.validation_errors)
            message = f"{message} Errors: [{details}]"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE


class SystemIntegrationInitializationException(IntegrationException):
    """Raised when EngineIntegrator fails to initialize components or register adapters."""

    ERROR_CODE = "ENG-343"

    def __init__(
        self,
        message: str = "EngineIntegrator initialization failed.",
        failed_adapters: Optional[List[str]] = None,
        **kwargs,
    ) -> None:
        self.failed_adapters = failed_adapters or []
        if self.failed_adapters:
            message = f"{message} Failed adapters: {self.failed_adapters}"
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE
