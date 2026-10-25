"""
ood_detection/__init__.py
=========================
Out-of-Distribution (OOD) Detection package for V2.0 platform.
Modular framework supporting MSP, Predictive Entropy, Energy Score,
Mahalanobis Distance, and KNN Feature Distance.
"""

from ood_detection.config import (
    SCORE_ORIENTATION_HIGHER_IS_OOD,
    MODALITY_CONFIGS,
    SEED,
)
from ood_detection.msp import compute_msp_score
from ood_detection.entropy import compute_predictive_entropy_score
from ood_detection.energy import compute_energy_score
from ood_detection.feature_distance import (
    MahalanobisDetector,
    KNNDetector,
)
KNNFeatureDetector = KNNDetector
from ood_detection.scoring import OODScorer
from ood_detection.thresholding import OODThresholdSelector
from ood_detection.evaluator import OODEvaluator
from ood_detection.modality_adapter import OODModalityAdapter

__all__ = [
    "SCORE_ORIENTATION_HIGHER_IS_OOD",
    "MODALITY_CONFIGS",
    "SEED",
    "compute_msp_score",
    "compute_predictive_entropy_score",
    "compute_energy_score",
    "MahalanobisDetector",
    "KNNFeatureDetector",
    "OODScorer",
    "OODThresholdSelector",
    "OODEvaluator",
    "OODModalityAdapter",
]
