"""
ood_detection/scoring.py
========================
Unified OOD scoring interface coordinating all 5 detection methods:
1. Maximum Softmax Probability (MSP complement)
2. Predictive Entropy
3. Free Energy Score
4. Class-Conditional Mahalanobis Distance
5. K-Nearest Neighbors (KNN) Feature Distance

Guarantees consistent score orientation across all methods:
HIGHER score = MORE OOD / ANOMALOUS.
LOWER score = MORE IN-DISTRIBUTION.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from ood_detection.config import SUPPORTED_METHODS
from ood_detection.energy import compute_energy_score
from ood_detection.entropy import compute_entropy_score
from ood_detection.feature_distance import KNNDetector, MahalanobisDetector
from ood_detection.msp import compute_msp_score

logger = logging.getLogger(__name__)


class OODScorer:
    """
    Unified dispatcher for OOD scoring methods.
    Supports multiclass, binary, and multi-label (Retina) architectures.
    """

    def __init__(
        self,
        mahalanobis_detector: Optional[MahalanobisDetector] = None,
        knn_detector: Optional[KNNDetector] = None,
    ):
        self.mahalanobis_detector = mahalanobis_detector
        self.knn_detector = knn_detector

    def compute_score(
        self,
        method: str,
        logits: Optional[torch.Tensor | np.ndarray] = None,
        features: Optional[np.ndarray | torch.Tensor] = None,
        task: str = "multiclass",
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        Computes OOD score for a batch of predictions or features.

        Args:
            method: One of ['msp', 'entropy', 'energy', 'mahalanobis', 'knn'].
            logits: Logit array/tensor, shape (N, C). Required for msp, entropy, energy.
            features: Feature embeddings, shape (N, D). Required for mahalanobis, knn.
            task: 'multiclass', 'binary', or 'multilabel'.
            temperature: Temperature parameter for msp, entropy, energy.

        Returns:
            1D numpy array of shape (N,) where HIGHER = MORE OOD.
        """
        method = method.lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError(
                f"Unsupported OOD method '{method}'. Supported: {SUPPORTED_METHODS}"
            )

        if method == "msp":
            if logits is None:
                raise ValueError("Logits must be provided for MSP scoring.")
            scores = compute_msp_score(logits, task=task, temperature=temperature)

        elif method == "entropy":
            if logits is None:
                raise ValueError("Logits must be provided for Entropy scoring.")
            scores = compute_entropy_score(logits, task=task, temperature=temperature)

        elif method == "energy":
            if logits is None:
                raise ValueError("Logits must be provided for Energy scoring.")
            scores = compute_energy_score(logits, task=task, temperature=temperature)

        elif method == "mahalanobis":
            if features is None:
                raise ValueError("Features must be provided for Mahalanobis scoring.")
            if self.mahalanobis_detector is None or not self.mahalanobis_detector.is_fitted:
                raise RuntimeError("Mahalanobis detector is not fitted.")
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            scores = self.mahalanobis_detector.score(features_np)

        elif method == "knn":
            if features is None:
                raise ValueError("Features must be provided for KNN scoring.")
            if self.knn_detector is None or not self.knn_detector.is_fitted:
                raise RuntimeError("KNN detector is not fitted.")
            if isinstance(features, torch.Tensor):
                features_np = features.detach().cpu().numpy()
            else:
                features_np = np.asarray(features)
            scores = self.knn_detector.score(features_np)

        else:
            raise ValueError(f"Unhandled method '{method}'")

        # Sanity check: Ensure finite scores
        scores = np.asarray(scores, dtype=np.float64)
        if not np.all(np.isfinite(scores)):
            logger.warning(f"Non-finite values encountered in {method} scores. Sanitizing.")
            # Replace non-finite with maximum valid score
            finite_max = np.nanmax(scores[np.isfinite(scores)]) if np.any(np.isfinite(scores)) else 1.0
            scores = np.nan_to_num(scores, nan=finite_max, posinf=finite_max, neginf=0.0)

        return scores
