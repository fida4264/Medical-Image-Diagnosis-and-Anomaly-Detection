"""
distance_calculator.py — Module 16.3: Confidence & OOD Detection Strategies
Medical AI Platform — Distance metric calculator (Mahalanobis, Euclidean, Cosine).
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple

from .strategy_exceptions import DistanceComputationError


class DistanceCalculator:
    """
    Computes distance metrics in feature space.

    Metrics
    -------
    Mahalanobis Distance:  D_M(z, mu) = sqrt( (z - mu)^T * Sigma^{-1} * (z - mu) )
    Euclidean Distance:    D_E(z, mu) = || z - mu ||_2
    Cosine Distance:       D_C(z, mu) = 1 - (z . mu) / (||z|| * ||mu||)
    """

    @staticmethod
    def compute_mahalanobis(
        feature_vec: np.ndarray,
        class_means: Dict[str, np.ndarray],
        precision_matrix: np.ndarray,
    ) -> Tuple[float, str]:
        """
        Compute the minimum Mahalanobis distance to any in-distribution class mean.

        Parameters
        ----------
        feature_vec : np.ndarray
            1D feature array (D,).
        class_means : Dict[str, np.ndarray]
            Map of class_name → mean array (D,).
        precision_matrix : np.ndarray
            Inverse covariance matrix (D, D).

        Returns
        -------
        Tuple[float, str]
            (min_mahalanobis_distance, nearest_class_name)
        """
        if not class_means or precision_matrix is None:
            raise DistanceComputationError("Class means or precision matrix not initialized.")

        z = np.asarray(feature_vec, dtype=np.float64).ravel()
        min_dist = float("inf")
        nearest_class = "unknown"

        for cls_name, mu in class_means.items():
            diff = z - np.asarray(mu, dtype=np.float64).ravel()
            # d_sq = diff^T * Precision * diff
            d_sq = float(np.dot(np.dot(diff, precision_matrix), diff))
            d_sq = max(d_sq, 0.0)  # Numerical floor for non-negative sqrt
            dist = float(np.sqrt(d_sq))

            if dist < min_dist:
                min_dist = dist
                nearest_class = cls_name

        return min_dist, nearest_class

    @staticmethod
    def compute_euclidean(feature_vec: np.ndarray, target_vec: np.ndarray) -> float:
        z = np.asarray(feature_vec, dtype=np.float64).ravel()
        t = np.asarray(target_vec, dtype=np.float64).ravel()
        return float(np.linalg.norm(z - t))

    @staticmethod
    def compute_cosine_distance(feature_vec: np.ndarray, target_vec: np.ndarray) -> float:
        z = np.asarray(feature_vec, dtype=np.float64).ravel()
        t = np.asarray(target_vec, dtype=np.float64).ravel()
        norm_z = np.linalg.norm(z)
        norm_t = np.linalg.norm(t)
        if norm_z == 0 or norm_t == 0:
            return 1.0
        similarity = float(np.dot(z, t) / (norm_z * norm_t))
        return float(1.0 - max(min(similarity, 1.0), -1.0))


__all__ = ["DistanceCalculator"]
