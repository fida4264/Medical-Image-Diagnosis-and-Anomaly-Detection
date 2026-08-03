"""
__init__.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Package exports for the Anomaly Detection Framework.
"""

from __future__ import annotations

from .anomaly_configuration import AnomalyConfiguration
from .anomaly_context import AnomalyContext
from .anomaly_detection_engine import AnomalyDetectionEngine
from .anomaly_exceptions import (
    AnomalyConfigurationError,
    AnomalyContextBuildError,
    AnomalyDetectionError,
    AnomalyResultAttachmentError,
    ConfidenceAnalysisError,
    DecisionEngineError,
    DetectionStrategyError,
    OODDetectionError,
    RegistryError,
    RiskAssessmentError,
    StrategyNotFoundError,
    UnknownDiseaseDetectionError,
)
from .anomaly_registry import AnomalyRegistry
from .anomaly_result import AnomalyResult, AnomalyStatus, RecommendedAction, RiskLevel
from .confidence_analyzer import ConfidenceAnalyzer
from .decision_engine import DecisionEngine
from .interfaces import IAnomalyRegistry, IDetectionStrategy
from .ood_detector import OODDetector
from .risk_assessor import RiskAssessor
from .unknown_disease_detector import UnknownDiseaseDetector

__all__ = [
    # Engine
    "AnomalyDetectionEngine",
    # Registry
    "AnomalyRegistry",
    # Configuration
    "AnomalyConfiguration",
    # Context
    "AnomalyContext",
    # Result & Enums
    "AnomalyResult",
    "AnomalyStatus",
    "RiskLevel",
    "RecommendedAction",
    # Sub-components
    "ConfidenceAnalyzer",
    "OODDetector",
    "UnknownDiseaseDetector",
    "RiskAssessor",
    "DecisionEngine",
    # Interfaces
    "IDetectionStrategy",
    "IAnomalyRegistry",
    # Exceptions
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
