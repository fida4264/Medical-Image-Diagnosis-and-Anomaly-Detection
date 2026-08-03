"""
chest_xray/training/metrics.py
================================
Metrics for Chest X-ray disease classification.
Re-exports compute_classification_metrics from Module 6 and adds weighted F1.
"""

from __future__ import annotations

from typing import List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from modality_classifier.training.metrics import compute_classification_metrics  # REUSE Module 6


def compute_weighted_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Computes weighted F1-score for class-imbalanced Chest X-ray datasets."""
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(f1_weighted)


__all__ = ["compute_classification_metrics", "compute_weighted_f1"]
