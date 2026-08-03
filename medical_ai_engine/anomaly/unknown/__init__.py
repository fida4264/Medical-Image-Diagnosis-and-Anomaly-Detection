"""
__init__.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Package exports for Unknown Disease Detection subsystem.
"""

from __future__ import annotations

from .feature_embedding_manager import EmbeddingRecord, FeatureEmbeddingManager
from .novelty_estimator import NoveltyEstimate, NoveltyEstimator
from .prototype_manager import ClassPrototype, PrototypeManager
from .similarity_engine import SimilarityEngine, SimilarityMetric, SimilarityResult
from .unknown_configuration import UnknownConfiguration
from .unknown_detector import UnknownDiseaseDetector
from .unknown_exceptions import (
    EmbeddingManagerError,
    NoveltyEstimatorError,
    PrototypeManagerError,
    SimilarityEngineError,
    UnknownConfigurationError,
    UnknownDetectionError,
)

__all__ = [
    "FeatureEmbeddingManager",
    "EmbeddingRecord",
    "PrototypeManager",
    "ClassPrototype",
    "SimilarityEngine",
    "SimilarityMetric",
    "SimilarityResult",
    "NoveltyEstimator",
    "NoveltyEstimate",
    "UnknownConfiguration",
    "UnknownDiseaseDetector",
    "UnknownDetectionError",
    "EmbeddingManagerError",
    "PrototypeManagerError",
    "SimilarityEngineError",
    "NoveltyEstimatorError",
    "UnknownConfigurationError",
]
