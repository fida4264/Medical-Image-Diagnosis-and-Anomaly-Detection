"""
ood_detection/feature_distance.py
=================================
Feature-space distance metrics for OOD detection:
1. Class-Conditional Mahalanobis Distance (Lee et al., NeurIPS 2018)
2. K-Nearest Neighbors (KNN) Feature Distance (Sun et al., ICML 2022)

Both methods fit their statistics (means, covariance, feature bank)
exclusively on permitted TRAINING features. Validation/test data are NEVER
used to estimate feature distributions.

Orientation: HIGHER score = FARTHER from training distribution = MORE OOD.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MahalanobisDetector:
    """
    Class-conditional Gaussian discriminant analysis on latent feature embeddings.
    Reference: Lee et al., "A Simple Unified Framework for Detecting
    Out-of-Distribution Samples and Adversarial Attacks", NeurIPS 2018.

    Fitted strictly on training features.
    """

    def __init__(self, reg_epsilon: float = 1e-4):
        self.reg_epsilon = reg_epsilon
        self.class_means: Dict[int, np.ndarray] = {}
        self.global_mean: Optional[np.ndarray] = None
        self.precision_matrix: Optional[np.ndarray] = None
        self.feature_dim: int = 512
        self.num_classes: int = 0
        self.is_fitted: bool = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        task: str = "multiclass",
    ) -> "MahalanobisDetector":
        """
        Fits class centroids and tied precision matrix on training data only.

        Args:
            features: Feature embeddings, shape (N_train, D).
            labels: Integer class indices (N_train,) or binary matrix (N_train, C) for multi-label.
            task: 'multiclass', 'binary', or 'multilabel'.

        Returns:
            self
        """
        N, D = features.shape
        self.feature_dim = D
        self.global_mean = np.mean(features, axis=0)

        if task == "multilabel":
            # Multi-label Retina: compute global mean and active-label centroids
            self.num_classes = labels.shape[1] if labels.ndim > 1 else 8
            # Compute centroid for normal and each major label
            self.class_means = {}
            centered_features = []
            for c in range(self.num_classes):
                mask = labels[:, c] == 1 if labels.ndim > 1 else labels == c
                if mask.sum() > 0:
                    c_mean = np.mean(features[mask], axis=0)
                    self.class_means[c] = c_mean
                    centered_features.append(features[mask] - c_mean)
                else:
                    self.class_means[c] = self.global_mean

            if centered_features:
                centered = np.vstack(centered_features)
            else:
                centered = features - self.global_mean

            cov = np.cov(centered, rowvar=False) + np.eye(D) * self.reg_epsilon
        else:
            # Multiclass or binary
            unique_classes = np.unique(labels.astype(int))
            self.num_classes = len(unique_classes)
            self.class_means = {}

            centered_list = []
            for c in unique_classes:
                idx = np.where(labels == c)[0]
                if len(idx) > 0:
                    c_mean = np.mean(features[idx], axis=0)
                    self.class_means[int(c)] = c_mean
                    centered_list.append(features[idx] - c_mean)
                else:
                    self.class_means[int(c)] = self.global_mean

            if centered_list:
                centered = np.vstack(centered_list)
            else:
                centered = features - self.global_mean

            cov = np.cov(centered, rowvar=False) + np.eye(D) * self.reg_epsilon

        # Precision matrix is inverse of tied covariance
        try:
            self.precision_matrix = np.linalg.pinv(cov)
        except Exception as e:
            logger.warning(f"Covariance inversion fallback: {e}")
            self.precision_matrix = np.linalg.inv(cov + np.eye(D) * 1e-2)

        self.is_fitted = True
        logger.info(f"MahalanobisDetector fitted on N={N}, D={D}, classes={len(self.class_means)}")
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """
        Calculates minimum Mahalanobis distance to any class centroid.

        Args:
            features: Query embeddings, shape (N, D).

        Returns:
            1D array of shape (N,) containing Mahalanobis distances.
            Higher distance indicates farther from in-distribution -> MORE OOD.
        """
        if not self.is_fitted:
            raise RuntimeError("MahalanobisDetector must be fitted before scoring.")

        N, D = features.shape
        min_distances = np.full(N, np.inf)

        for c, mu_c in self.class_means.items():
            diff = features - mu_c  # (N, D)
            # Mahalanobis distance: sqrt(sum(diff @ prec * diff, axis=1))
            dist_sq = np.sum(np.dot(diff, self.precision_matrix) * diff, axis=1)
            # Clip negative values due to numerical precision
            dist_sq = np.maximum(dist_sq, 0.0)
            dist = np.sqrt(dist_sq)
            min_distances = np.minimum(min_distances, dist)

        return min_distances.astype(np.float64)

    def save(self, path: Path | str) -> None:
        """Saves fitted parameters to an npz file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "feature_dim": np.array([self.feature_dim]),
            "num_classes": np.array([self.num_classes]),
            "global_mean": self.global_mean,
            "precision_matrix": self.precision_matrix,
        }
        for c, mu in self.class_means.items():
            data[f"class_mean_{c}"] = mu
        np.savez_compressed(path, **data)
        logger.info(f"Saved MahalanobisDetector to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "MahalanobisDetector":
        """Loads fitted parameters from an npz file."""
        path = Path(path)
        detector = cls()
        loaded = np.load(path)
        detector.feature_dim = int(loaded["feature_dim"][0])
        detector.num_classes = int(loaded["num_classes"][0])
        detector.global_mean = loaded["global_mean"]
        detector.precision_matrix = loaded["precision_matrix"]
        detector.class_means = {}
        for key in loaded.files:
            if key.startswith("class_mean_"):
                c = int(key.replace("class_mean_", ""))
                detector.class_means[c] = loaded[key]
        detector.is_fitted = True
        return detector


class KNNDetector:
    """
    K-Nearest Neighbors (KNN) feature-space distance OOD detector.
    Reference: Sun et al., "Out-of-Distribution Detection with Deep
    Nearest Neighbors", ICML 2022.

    Computes normalized Euclidean/cosine distance to the k-nearest training
    embeddings. Non-parametric, requires no covariance estimation.
    """

    def __init__(self, k: int = 5, max_bank_size: int = 1500):
        self.k = k
        self.max_bank_size = max_bank_size
        self.memory_bank: Optional[np.ndarray] = None
        self.is_fitted: bool = False

    def fit(self, features: np.ndarray, labels: Optional[np.ndarray] = None) -> "KNNDetector":
        """
        Stores (subsampled) normalized training embeddings into memory bank.

        Args:
            features: Training embeddings, shape (N_train, D).
            labels: Optional training labels.

        Returns:
            self
        """
        N = len(features)
        # Normalize features to unit sphere (cosine distance geometry)
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normed = features / norms

        if N > self.max_bank_size:
            # Subsample deterministically with seed 42 to keep inference fast
            rng = np.random.RandomState(42)
            idx = rng.choice(N, size=self.max_bank_size, replace=False)
            self.memory_bank = normed[idx].astype(np.float32)
        else:
            self.memory_bank = normed.astype(np.float32)

        self.is_fitted = True
        logger.info(f"KNNDetector fitted with memory bank size: {len(self.memory_bank)}")
        return self

    def score(self, features: np.ndarray, k: Optional[int] = None) -> np.ndarray:
        """
        Calculates average distance to the k-nearest neighbors in memory bank.

        Args:
            features: Query embeddings, shape (N, D).
            k: Number of nearest neighbors (defaults to self.k).

        Returns:
            1D array of shape (N,) containing KNN distances.
            Higher distance indicates farther from training bank -> MORE OOD.
        """
        if not self.is_fitted or self.memory_bank is None:
            raise RuntimeError("KNNDetector must be fitted before scoring.")

        k_val = k if k is not None else self.k
        k_val = min(k_val, len(self.memory_bank))

        # Normalize query features
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-10
        normed_queries = features / norms

        # Cosine distance = 1.0 - Cosine similarity = 1.0 - (Q @ B.T)
        sim_matrix = np.dot(normed_queries, self.memory_bank.T)  # (N, N_bank)
        dist_matrix = 1.0 - sim_matrix
        dist_matrix = np.maximum(dist_matrix, 0.0)

        # Sort along bank axis and take top k smallest distances
        topk_distances = np.partition(dist_matrix, k_val, axis=1)[:, :k_val]
        avg_knn_dist = np.mean(topk_distances, axis=1)

        return avg_knn_dist.astype(np.float64)

    def save(self, path: Path | str) -> None:
        """Saves memory bank to npz."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, memory_bank=self.memory_bank, k=self.k)
        logger.info(f"Saved KNNDetector to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "KNNDetector":
        """Loads memory bank from npz."""
        path = Path(path)
        loaded = np.load(path)
        detector = cls(k=int(loaded["k"]))
        detector.memory_bank = loaded["memory_bank"]
        detector.is_fitted = True
        return detector
