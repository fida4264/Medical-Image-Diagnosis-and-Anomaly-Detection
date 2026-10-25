"""
anomaly_detection/__init__.py
=============================
Anomaly Detection package for V2.0 platform.
Separates within-distribution reference abnormality scoring from OOD detection.
"""

from anomaly_detection.evaluator import AnomalyEvaluator
from anomaly_detection.scoring import ReferenceDistributionAnomalyScorer

__all__ = [
    "ReferenceDistributionAnomalyScorer",
    "AnomalyEvaluator",
]
