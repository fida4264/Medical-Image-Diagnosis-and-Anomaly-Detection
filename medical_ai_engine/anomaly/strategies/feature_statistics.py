"""
feature_statistics.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — In-distribution Gaussian class statistics calculator (means & covariance).
"""

from __future__ import annotations

import logging
import numpy as np
from typing import Dict, Tuple

from .strategy_exceptions import StatisticsComputationError

logger = logging.getLogger(__name__)


class FeatureStatistics:
    """
    Computes and maintains class mean vectors mu_c and regularized covariance matrix Sigma.

    Mathematics
    -----------
    Class Mean:  mu_c = (1 / N_c) * sum_{i=1}^{N_c} z_i
    Shared Covariance: Sigma = (1 / N) * sum_{c} sum_{i} (z_i - mu_c)(z_i - mu_c)^T
    Regularized Covariance: Sigma_reg = Sigma + reg * I
    """

    def __init__(self, regularization: float = 1e-4) -> None:
        self.regularization = regularization
        self.class_means: Dict[str, np.ndarray] = {}
        self.precision_matrix: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, features_by_class: Dict[str, np.ndarray]) -> None:
        """
        Fit class means and shared precision (inverse covariance) matrix from in-distribution training features.

        Parameters
        ----------
        features_by_class : Dict[str, np.ndarray]
            Map of class_name → array of shape (N_c, D).
        """
        try:
            all_centered = []
            self.class_means.clear()

            for cls_name, feats in features_by_class.items():
                feats = np.asarray(feats, dtype=np.float32)
                if feats.ndim == 1:
                    feats = feats.reshape(1, -1)
                mean = np.mean(feats, axis=0)
                self.class_means[cls_name] = mean
                centered = feats - mean
                all_centered.append(centered)

            all_centered_arr = np.vstack(all_centered)
            N, D = all_centered_arr.shape

            # Compute shared empirical covariance
            cov = np.cov(all_centered_arr, rowvar=False)
            if D == 1:
                cov = np.array([[cov]])

            # Add regularization for invertibility (Ledoit-Wolf / Tikhonov)
            cov_reg = cov + np.eye(D, dtype=np.float32) * self.regularization

            # Compute precision matrix (inverse covariance)
            self.precision_matrix = np.linalg.pinv(cov_reg)
            self.is_fitted = True
            logger.info("[FeatureStatistics] Fitted statistics for %d classes (feature dim D=%d).", len(self.class_means), D)

        except Exception as exc:
            self.is_fitted = False
            raise StatisticsComputationError(f"Failed to fit feature statistics: {exc}") from exc

    def fit_synthetic_defaults(self, class_names: list[str], feature_dim: int = 512) -> None:
        """Fit deterministic synthetic mean and identity covariance for testing/stubs."""
        features_by_class = {}
        for idx, cls in enumerate(class_names):
            rng = np.random.RandomState(idx + 100)
            # Distinct cluster center for each class
            center = rng.randn(feature_dim).astype(np.float32) * 2.0
            samples = center + rng.randn(20, feature_dim).astype(np.float32) * 0.2
            features_by_class[cls] = samples
        self.fit(features_by_class)


__all__ = ["FeatureStatistics"]
