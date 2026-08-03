"""
framework_exceptions.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Typed exception hierarchy for the anomaly execution framework.
"""

from __future__ import annotations

from medical_ai_engine.anomaly.anomaly_exceptions import AnomalyDetectionError


class AnomalyFrameworkError(AnomalyDetectionError):
    """Base exception for all anomaly detection framework failures."""


class FrameworkConfigurationError(AnomalyFrameworkError):
    """Raised when framework configuration parameters are invalid."""


class PluginLoadingError(AnomalyFrameworkError):
    """Raised when dynamic loading of a strategy plugin fails."""


class PluginValidationError(AnomalyFrameworkError):
    """Raised when a strategy plugin fails validation checks."""


class StrategyExecutionError(AnomalyFrameworkError):
    """Raised when strategy execution encounters a runtime failure."""


class PipelineExecutionError(AnomalyFrameworkError):
    """Raised when the execution pipeline fails."""


class CacheError(AnomalyFrameworkError):
    """Raised when cache operations fail."""


__all__ = [
    "AnomalyFrameworkError",
    "FrameworkConfigurationError",
    "PluginLoadingError",
    "PluginValidationError",
    "StrategyExecutionError",
    "PipelineExecutionError",
    "CacheError",
]
