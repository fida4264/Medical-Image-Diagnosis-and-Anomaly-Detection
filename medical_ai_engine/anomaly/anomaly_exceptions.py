"""
anomaly_exceptions.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Typed exception hierarchy for the Anomaly Detection Framework.
"""

from __future__ import annotations


class AnomalyDetectionError(Exception):
    """Base exception for all Anomaly Detection Framework failures."""

    def __init__(self, message: str, request_id: str = "", details: dict | None = None) -> None:
        super().__init__(message)
        self.request_id = request_id
        self.details: dict = details or {}

    def __str__(self) -> str:
        base = super().__str__()
        rid = f" [request_id={self.request_id}]" if self.request_id else ""
        return f"{base}{rid}"


class AnomalyConfigurationError(AnomalyDetectionError):
    """Raised when the anomaly detection configuration is invalid."""


class AnomalyContextBuildError(AnomalyDetectionError):
    """Raised when AnomalyContext cannot be constructed from prediction inputs."""


class DetectionStrategyError(AnomalyDetectionError):
    """Raised when a detection strategy encounters a runtime failure."""


class StrategyNotFoundError(AnomalyDetectionError):
    """Raised when a requested detection strategy is not registered."""


class RegistryError(AnomalyDetectionError):
    """Raised when the AnomalyRegistry encounters a registration or lookup failure."""


class ConfidenceAnalysisError(AnomalyDetectionError):
    """Raised when confidence analysis fails unexpectedly."""


class OODDetectionError(AnomalyDetectionError):
    """Raised when out-of-distribution detection fails."""


class UnknownDiseaseDetectionError(AnomalyDetectionError):
    """Raised when unknown disease detection fails."""


class RiskAssessmentError(AnomalyDetectionError):
    """Raised when risk level evaluation fails."""


class DecisionEngineError(AnomalyDetectionError):
    """Raised when the decision engine cannot produce a recommendation."""


class AnomalyResultAttachmentError(AnomalyDetectionError):
    """Raised when AnomalyResult cannot be attached to PredictionResult."""


__all__ = [
    "AnomalyDetectionError",
    "AnomalyConfigurationError",
    "AnomalyContextBuildError",
    "DetectionStrategyError",
    "StrategyNotFoundError",
    "RegistryError",
    "ConfidenceAnalysisError",
    "OODDetectionError",
    "UnknownDiseaseDetectionError",
    "RiskAssessmentError",
    "DecisionEngineError",
    "AnomalyResultAttachmentError",
]
