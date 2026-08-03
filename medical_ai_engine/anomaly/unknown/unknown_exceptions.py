"""
unknown_exceptions.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Typed exception hierarchy.
"""
from __future__ import annotations
from medical_ai_engine.anomaly.anomaly_exceptions import AnomalyDetectionError


class UnknownDetectionError(AnomalyDetectionError):
    """Base exception for the Unknown Disease Detection subsystem."""


class EmbeddingManagerError(UnknownDetectionError):
    """Raised when feature embedding management fails."""


class PrototypeManagerError(UnknownDetectionError):
    """Raised when prototype generation or lookup fails."""


class SimilarityEngineError(UnknownDetectionError):
    """Raised when similarity computation fails."""


class NoveltyEstimatorError(UnknownDetectionError):
    """Raised when novelty estimation encounters invalid state."""


class UnknownConfigurationError(UnknownDetectionError):
    """Raised when configuration is invalid."""


__all__ = [
    "UnknownDetectionError",
    "EmbeddingManagerError",
    "PrototypeManagerError",
    "SimilarityEngineError",
    "NoveltyEstimatorError",
    "UnknownConfigurationError",
]
