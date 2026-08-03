"""
retina/training/metrics.py
============================
Metrics for Retina disease classification.
Re-exports compute_classification_metrics from Module 6.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from modality_classifier.training.metrics import compute_classification_metrics  # Module 6 reuse


def compute_weighted_f1(y_true: np.ndarray | List[int], y_pred: np.ndarray) -> float:
    """Computes weighted F1-score for class-imbalanced datasets."""
    _, _, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return float(f1_weighted)
