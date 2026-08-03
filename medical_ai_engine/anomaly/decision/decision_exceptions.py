"""
decision_exceptions.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Exception hierarchy for the Decision Fusion subsystem.
"""
from __future__ import annotations
from medical_ai_engine.anomaly.anomaly_exceptions import AnomalyDetectionError


class DecisionError(AnomalyDetectionError):
    """Base exception for the Risk Assessment & Decision Fusion subsystem."""


class FusionEngineError(DecisionError):
    """Raised when decision signal fusion fails."""


class PolicyError(DecisionError):
    """Raised when a decision policy fails or receives invalid configuration."""


class PriorityError(DecisionError):
    """Raised when signal priority resolution encounters an error."""


class DecisionConfigurationError(DecisionError):
    """Raised when decision fusion configuration is invalid."""


__all__ = [
    "DecisionError",
    "FusionEngineError",
    "PolicyError",
    "PriorityError",
    "DecisionConfigurationError",
]
