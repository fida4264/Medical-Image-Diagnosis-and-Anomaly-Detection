"""
__init__.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Strategies package exports.
"""

from __future__ import annotations

from .confidence_threshold import ConfidenceThresholdStrategy
from .distance_calculator import DistanceCalculator
from .feature_extractor import FeatureExtractor
from .feature_statistics import FeatureStatistics
from .mahalanobis_detector import MahalanobisOODStrategy
from .strategy_configuration import ModalityThresholds, StrategyConfiguration
from .strategy_exceptions import (
    DistanceComputationError,
    FeatureExtractionError,
    StatisticsComputationError,
    StrategyError,
    ThresholdManagerError,
)
from .threshold_manager import ThresholdManager

__all__ = [
    "ConfidenceThresholdStrategy",
    "MahalanobisOODStrategy",
    "FeatureExtractor",
    "FeatureStatistics",
    "DistanceCalculator",
    "ThresholdManager",
    "StrategyConfiguration",
    "ModalityThresholds",
    "StrategyError",
    "FeatureExtractionError",
    "StatisticsComputationError",
    "DistanceComputationError",
    "ThresholdManagerError",
]
