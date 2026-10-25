"""
training/v2_baseline/metrics.py
===============================
Modality-specific metric calculation for V2.0 baseline evaluations.
Provides multiclass, binary, and multi-label metrics with safe edge-case handling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    modality: str = "general",
) -> Dict[str, Any]:
    """
    Computes comprehensive multiclass/binary classification metrics.

    Args:
        y_true: 1D array of ground truth class integer indices.
        y_pred: 1D array of predicted class integer indices.
        y_prob: 2D array of predicted class probabilities (N, num_classes).
        class_names: List of class names.
        modality: Modality identifier for special metric reporting.

    Returns:
        Dictionary of computed metrics, per-class details, and confusion matrix.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    num_classes = len(class_names) if class_names else int(np.max(y_true) + 1)

    # 1. Standard global metrics
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    # 2. Per-class Precision, Recall, F1, Support
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )

    per_class = {}
    for i in range(num_classes):
        c_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
        per_class[c_name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    # 3. Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    cm_list = cm.tolist()

    # 4. ROC-AUC calculation
    macro_auc = None
    per_class_auc = {}
    if y_prob is not None and y_prob.shape[0] == len(y_true):
        try:
            if num_classes == 2:
                # Binary: y_prob is (N, 2), take column 1 (positive class)
                pos_prob = y_prob[:, 1] if y_prob.ndim == 2 else y_prob
                macro_auc = float(roc_auc_score(y_true, pos_prob))
            else:
                # Multiclass: one-vs-rest
                macro_auc = float(
                    roc_auc_score(
                        y_true,
                        y_prob,
                        multi_class="ovr",
                        average="macro",
                        labels=list(range(num_classes)),
                    )
                )
                for i in range(num_classes):
                    c_name = class_names[i] if class_names and i < len(class_names) else f"class_{i}"
                    y_true_binary = (y_true == i).astype(int)
                    if len(np.unique(y_true_binary)) > 1:
                        per_class_auc[c_name] = float(roc_auc_score(y_true_binary, y_prob[:, i]))
                    else:
                        per_class_auc[c_name] = None
        except Exception as e:
            logger.warning(f"ROC-AUC computation failed: {e}")
            macro_auc = None

    metrics: Dict[str, Any] = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": balanced_acc,
        "macro_auc": macro_auc,
        "per_class": per_class,
        "per_class_auc": per_class_auc if per_class_auc else None,
        "confusion_matrix": cm_list,
        "num_samples": len(y_true),
    }

    # 5. Modality-specific additions (e.g. Brain CT sensitivity / specificity)
    if modality == "brain_ct" and num_classes == 2:
        # Class 0: normal, Class 1: hemorrhage
        # True positives = cm[1, 1], False negatives = cm[1, 0]
        # True negatives = cm[0, 0], False positives = cm[0, 1]
        tn = int(cm[0, 0])
        fp = int(cm[0, 1])
        fn = int(cm[1, 0])
        tp = int(cm[1, 1])
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = specificity
        metrics["binary_tp"] = tp
        metrics["binary_tn"] = tn
        metrics["binary_fp"] = fp
        metrics["binary_fn"] = fn

    return metrics


def calculate_multilabel_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes multi-label classification metrics for Retina (8 binary targets).

    Args:
        y_true: 2D binary matrix (N, num_classes) of ground truth.
        y_prob: 2D probability matrix (N, num_classes) from sigmoid.
        threshold: Decision threshold for positive classification (default 0.5).
        class_names: Names of the multi-label targets.

    Returns:
        Dictionary of overall macro/micro metrics and per-label statistics.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= threshold).astype(np.int64)

    num_samples, num_classes = y_true.shape
    if class_names is None:
        class_names = [f"label_{i}" for i in range(num_classes)]

    # Global multi-label F1 scores
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    micro_f1 = float(f1_score(y_true, y_pred, average="micro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Per-label metrics
    per_label: Dict[str, Dict[str, Any]] = {}
    label_aucs: List[float] = []

    for i, name in enumerate(class_names):
        yt_i = y_true[:, i]
        yp_i = y_pred[:, i]
        yprob_i = y_prob[:, i]

        p, r, f, s = precision_recall_fscore_support(yt_i, yp_i, average="binary", zero_division=0)

        auc = None
        if len(np.unique(yt_i)) > 1:
            try:
                auc = float(roc_auc_score(yt_i, yprob_i))
                label_aucs.append(auc)
            except Exception:
                auc = None

        tp = int(np.sum((yt_i == 1) & (yp_i == 1)))
        tn = int(np.sum((yt_i == 0) & (yp_i == 0)))
        fp = int(np.sum((yt_i == 0) & (yp_i == 1)))
        fn = int(np.sum((yt_i == 1) & (yp_i == 0)))

        per_label[name] = {
            "precision": float(p),
            "recall": float(r),
            "f1": float(f),
            "auc": auc,
            "support": int(np.sum(yt_i)),
            "tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
        }

    macro_auc = float(np.mean(label_aucs)) if label_aucs else None

    # Exact match ratio (subset accuracy)
    exact_match = float(np.mean(np.all(y_true == y_pred, axis=1)))

    return {
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "exact_match_ratio": exact_match,
        "per_label": per_label,
        "num_samples": num_samples,
    }
