"""
integration_exceptions.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Typed exception hierarchy for the integration subsystem.
"""

from __future__ import annotations


class ExplainabilityIntegrationError(Exception):
    """Base exception for Module 15.6 integration failures."""

    def __init__(self, message: str, request_id: str = "", details: dict | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.details: dict = details or {}

    def __str__(self) -> str:  # pragma: no cover
        base = super().__str__()
        rid = f" [request_id={self.request_id}]" if self.request_id else ""
        return f"{base}{rid}"


class IntegrationConfigurationError(ExplainabilityIntegrationError):
    """Raised when the integration configuration is invalid."""


class ContextBuildError(ExplainabilityIntegrationError):
    """Raised when ExplainabilityContext cannot be built from EngineContext."""


class ExecutionManagerError(ExplainabilityIntegrationError):
    """Raised when the explainability execution manager encounters a fatal error."""


class WorkflowCoordinationError(ExplainabilityIntegrationError):
    """Raised when the workflow coordinator cannot dispatch the pipeline."""


class EventListenerError(ExplainabilityIntegrationError):
    """Raised when the event listener encounters an unexpected error."""


class IntegrationValidationError(ExplainabilityIntegrationError):
    """Raised when integration output validation fails."""


class ResultAttachmentError(ExplainabilityIntegrationError):
    """Raised when attaching ExplainabilityResult to PredictionResult fails."""


__all__ = [
    "ExplainabilityIntegrationError",
    "IntegrationConfigurationError",
    "ContextBuildError",
    "ExecutionManagerError",
    "WorkflowCoordinationError",
    "EventListenerError",
    "IntegrationValidationError",
    "ResultAttachmentError",
]
