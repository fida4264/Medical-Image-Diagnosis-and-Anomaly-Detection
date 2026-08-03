"""
integration_exceptions.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Exception hierarchy for Anomaly Engine Integration.
"""
from __future__ import annotations
from medical_ai_engine.anomaly.anomaly_exceptions import AnomalyDetectionError


class AnomalyIntegrationError(AnomalyDetectionError):
    """Base exception for the Anomaly Engine Integration subsystem."""


class WorkflowCoordinationError(AnomalyIntegrationError):
    """Raised when post-prediction workflow coordination fails."""


class EventListenerError(AnomalyIntegrationError):
    """Raised when event listener processing fails."""


class ExecutionManagerError(AnomalyIntegrationError):
    """Raised when integration execution manager encounters an error."""


class IntegrationValidationError(AnomalyIntegrationError):
    """Raised when validation of inputs for integration fails."""


class IntegrationConfigurationError(AnomalyIntegrationError):
    """Raised when integration configuration is invalid."""


__all__ = [
    "AnomalyIntegrationError",
    "WorkflowCoordinationError",
    "EventListenerError",
    "ExecutionManagerError",
    "IntegrationValidationError",
    "IntegrationConfigurationError",
]
