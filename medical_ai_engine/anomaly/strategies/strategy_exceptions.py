"""
strategy_exceptions.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Typed exception hierarchy for anomaly detection strategies.
"""

from __future__ import annotations

from medical_ai_engine.anomaly.anomaly_exceptions import AnomalyDetectionError


class StrategyError(AnomalyDetectionError):
    """Base exception for strategy component failures."""


class FeatureExtractionError(StrategyError):
    """Raised when feature extraction from a model fails."""


class StatisticsComputationError(StrategyError):
    """Raised when covariance matrix or class mean computation fails."""


class DistanceComputationError(StrategyError):
    """Raised when distance metric calculation encounters singular matrix or shape errors."""


class ThresholdManagerError(StrategyError):
    """Raised when threshold lookup or configuration is invalid."""


__all__ = [
    "StrategyError",
    "FeatureExtractionError",
    "StatisticsComputationError",
    "DistanceComputationError",
    "ThresholdManagerError",
]
