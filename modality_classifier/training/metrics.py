"""
modality_classifier/training/metrics.py
========================================
Classification metrics computation (Accuracy, Top-1, Top-3, Precision, Recall, Macro F1).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def compute_classification_metrics(
    y_true: np.ndarray | List[int],
    y_pred_probs: np.ndarray,
    top_k: Tuple[int, ...] = (1, 3),
) -> Dict[str, float]:
    """
    Computes top-k accuracy, macro precision, recall, and F1 score.
    """
    y_true_arr = np.array(y_true, dtype=int)
    y_pred_arr = np.argmax(y_pred_probs, axis=1)

    acc = float(accuracy_score(y_true_arr, y_pred_arr))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true_arr, y_pred_arr, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": acc,
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
    }

    # Top-K Accuracies
    num_classes = y_pred_probs.shape[1]
    for k in top_k:
        if k > num_classes:
            continue
        top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
        correct = np.any(top_k_preds == y_true_arr[:, None], axis=1)
        metrics[f"top_{k}_accuracy"] = float(np.mean(correct))

    return metrics
