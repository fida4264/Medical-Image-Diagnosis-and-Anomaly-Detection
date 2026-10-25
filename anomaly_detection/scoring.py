"""
anomaly_detection/scoring.py
============================
Reference distribution anomaly scoring for medical images.

Distinct from Out-of-Distribution (OOD) Detection:
- OOD Question: "Does this input differ from the supported modality distribution?"
- Anomaly Question: "Does this in-distribution input show an unusual deviation relative to the normal reference distribution?"

Fits reference statistics strictly on confirmed normal/healthy TRAINING samples.
Score orientation: HIGHER score = GREATER deviation from normal reference.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ReferenceDistributionAnomalyScorer:
    """
    Measures deviation of in-distribution latent embeddings relative to a
    healthy/normal reference baseline distribution.
    """

    def __init__(self, reg_epsilon: float = 1e-4):
        self.reg_epsilon = reg_epsilon
        self.reference_mean: Optional[np.ndarray] = None
        self.reference_precision: Optional[np.ndarray] = None
        self.reference_class_name: str = "normal"
        self.is_fitted: bool = False

    def fit_reference(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        normal_class_idx: int,
        normal_class_name: str = "normal",
        task: str = "multiclass",
    ) -> "ReferenceDistributionAnomalyScorer":
        """
        Fits reference Gaussian parameters exclusively on normal training data.

        Args:
            features: Training embeddings, shape (N_train, D).
            labels: Training labels (integer indices or multi-label matrix).
            normal_class_idx: Column or class index representing 'normal' / 'healthy'.
            normal_class_name: Human-readable reference name.
            task: 'multiclass', 'binary', or 'multilabel'.

        Returns:
            self
        """
        if task == "multilabel":
            # For Retina: normal is when normal head == 1 and all disease heads == 0
            if labels.ndim > 1:
                mask = labels[:, normal_class_idx] == 1
            else:
                mask = labels == normal_class_idx
        else:
            if labels.ndim > 1:
                mask = labels[:, normal_class_idx] == 1
            else:
                mask = labels == normal_class_idx

        normal_features = features[mask]
        N_norm, D = normal_features.shape

        if N_norm < 10:
            logger.warning(
                f"Low sample count for normal reference ({N_norm}). Using global feature mean."
            )
            normal_features = features

        self.reference_class_name = normal_class_name
        self.reference_mean = np.mean(normal_features, axis=0)

        centered = normal_features - self.reference_mean
        cov = np.cov(centered, rowvar=False) + np.eye(D) * self.reg_epsilon

        try:
            self.reference_precision = np.linalg.pinv(cov)
        except Exception as e:
            logger.warning(f"Precision matrix fallback for reference anomaly: {e}")
            self.reference_precision = np.linalg.inv(cov + np.eye(D) * 1e-2)

        self.is_fitted = True
        logger.info(
            f"ReferenceDistributionAnomalyScorer fitted on {N_norm} normal samples ({normal_class_name})."
        )
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Calculates Mahalanobis distance from normal reference distribution.

        Args:
            features: Query latent embeddings, shape (N, D).

        Returns:
            1D array of anomaly scores. Higher = more anomalous / deviated from normal.
        """
        if not self.is_fitted or self.reference_mean is None or self.reference_precision is None:
            raise RuntimeError("Anomaly scorer must be fitted on reference data before scoring.")

        diff = features - self.reference_mean  # (N, D)
        dist_sq = np.sum(np.dot(diff, self.reference_precision) * diff, axis=1)
        dist_sq = np.maximum(dist_sq, 0.0)
        anomaly_scores = np.sqrt(dist_sq)

        return anomaly_scores.astype(np.float64)

    def save(self, path: Path | str) -> None:
        """Saves fitted reference statistics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            reference_mean=self.reference_mean,
            reference_precision=self.reference_precision,
            reference_class_name=self.reference_class_name,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ReferenceDistributionAnomalyScorer":
        """Loads fitted reference statistics."""
        path = Path(path)
        loaded = np.load(path)
        scorer = cls()
        scorer.reference_mean = loaded["reference_mean"]
        scorer.reference_precision = loaded["reference_precision"]
        scorer.reference_class_name = str(loaded["reference_class_name"])
        scorer.is_fitted = True
        return scorer
