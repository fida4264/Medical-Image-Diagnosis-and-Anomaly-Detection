"""
training/v2_baseline/loss_factory.py
====================================
Loss function factory with class-imbalance awareness.
Supports CrossEntropyLoss (unweighted & weighted) and BCEWithLogitsLoss (with positive weighting).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_class_weights(
    labels: Union[np.ndarray, List[int]],
    num_classes: int,
) -> torch.Tensor:
    """
    Computes balanced class weights for imbalanced classification:
    weight[c] = total_samples / (num_classes * count[c])
    """
    labels_arr = np.asarray(labels, dtype=np.int64)
    total_samples = len(labels_arr)
    weights = np.zeros(num_classes, dtype=np.float32)

    counts = np.bincount(labels_arr, minlength=num_classes)
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = total_samples / (num_classes * counts[c])
        else:
            weights[c] = 1.0

    # Normalize so mean weight is 1.0
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def compute_multilabel_pos_weights(
    multilabel_matrix: np.ndarray,
) -> torch.Tensor:
    """
    Computes per-label positive weights for BCEWithLogitsLoss:
    pos_weight[c] = (num_negatives) / (num_positives + eps)
    """
    total_samples = multilabel_matrix.shape[0]
    pos_counts = np.sum(multilabel_matrix, axis=0)
    neg_counts = total_samples - pos_counts

    pos_weights = (neg_counts + 1e-5) / (pos_counts + 1e-5)
    # Clip extreme positive weights to avoid gradient instability
    pos_weights = np.clip(pos_weights, 0.1, 20.0)
    return torch.tensor(pos_weights, dtype=torch.float32)


def create_loss_function(
    loss_type: str = "cross_entropy",
    task: str = "multiclass",
    num_classes: int = 4,
    class_weights: Optional[torch.Tensor] = None,
    pos_weights: Optional[torch.Tensor] = None,
    device: str = "cpu",
) -> nn.Module:
    """
    Instantiates PyTorch loss function configured with appropriate weights.
    """
    loss_type = loss_type.lower()
    task = task.lower()

    if task == "multilabel" or loss_type in ["bce_with_logits", "bce"]:
        if pos_weights is not None:
            pos_weights = pos_weights.to(device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)
            logger.info(f"Initialized BCEWithLogitsLoss with positive weights: {pos_weights.tolist()}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Initialized unweighted BCEWithLogitsLoss.")
        return criterion

    # Multiclass or binary
    if loss_type == "weighted_ce" or (loss_type == "cross_entropy" and class_weights is not None):
        if class_weights is not None:
            class_weights = class_weights.to(device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            logger.info(f"Initialized weighted CrossEntropyLoss with weights: {class_weights.tolist()}")
        else:
            criterion = nn.CrossEntropyLoss()
            logger.info("Initialized standard CrossEntropyLoss (no weights supplied).")
    else:
        criterion = nn.CrossEntropyLoss()
        logger.info("Initialized standard CrossEntropyLoss.")

    return criterion
