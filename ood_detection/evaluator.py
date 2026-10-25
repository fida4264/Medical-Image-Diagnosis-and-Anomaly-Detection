"""
ood_detection/evaluator.py
==========================
Evaluation metrics for Out-of-Distribution (OOD) detection.
Computes:
  - AUROC (In-Distribution vs OOD discrimination)
  - AUPR-OOD (Precision-Recall with OOD as positive class)
  - AUPR-In (Precision-Recall with ID as positive class)
  - FPR@95% TPR (False Positive Rate when 95% of OOD are detected, or 95% ID retention)
  - Operating Point Metrics (Sensitivity, Specificity, Precision, Recall, F1)
  - False Rejection Rate (FRR: fraction of ID incorrectly flagged as OOD)
  - False Acceptance Rate (FAR: fraction of OOD incorrectly accepted as ID)
  - Confusion Matrix [[TN, FP], [FN, TP]]
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    auc,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

logger = logging.getLogger(__name__)


def compute_fpr95(
    y_true: np.ndarray,
    y_scores: np.ndarray,
) -> float:
    """
    Computes the False Positive Rate (FPR) when True Positive Rate (TPR) is at least 95%.
    Positive class (1) = OOD, Negative class (0) = ID.
    Higher score indicates higher OOD likelihood.

    Args:
        y_true: Binary ground truth (0 for ID, 1 for OOD).
        y_scores: OOD scores.

    Returns:
        FPR at 95% TPR (float in [0, 1]).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    # Find smallest index where tpr >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) > 0:
        return float(fpr[idx[0]])
    return 1.0


def evaluate_ood_metrics(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    threshold: Optional[float] = None,
    modality: str = "unknown",
    method: str = "unknown",
    scenario: str = "unknown",
) -> Dict[str, Any]:
    """
    Evaluates comprehensive OOD detection performance.

    Positive class convention:
      Class 1 = OOD (Out-of-Distribution)
      Class 0 = In-Distribution (ID)
    Score orientation: HIGHER score = MORE OOD.

    Args:
        id_scores: 1D array of OOD scores for in-distribution samples.
        ood_scores: 1D array of OOD scores for out-of-distribution samples.
        threshold: Frozen decision threshold. If None, chosen via 95% ID retention.
        modality: Modality name for reporting.
        method: Scoring method name.
        scenario: OOD scenario identifier.

    Returns:
        Dictionary with all evaluated metrics and metadata.
    """
    id_s = np.asarray(id_scores, dtype=np.float64)
    ood_s = np.asarray(ood_scores, dtype=np.float64)

    n_id = len(id_s)
    n_ood = len(ood_s)

    if n_id == 0 or n_ood == 0:
        logger.warning(f"Empty evaluation array: n_id={n_id}, n_ood={n_ood}")
        return {
            "modality": modality,
            "method": method,
            "scenario": scenario,
            "n_id": n_id,
            "n_ood": n_ood,
            "status": "unavailable_insufficient_samples",
        }

    y_true = np.concatenate([np.zeros(n_id, dtype=int), np.ones(n_ood, dtype=int)])
    scores = np.concatenate([id_s, ood_s])

    # 1. AUROC (OOD is positive class)
    try:
        auroc = float(roc_auc_score(y_true, scores))
    except Exception as e:
        logger.warning(f"AUROC calculation error: {e}")
        auroc = float("nan")

    # 2. AUPR-OOD (OOD positive class)
    try:
        aupr_ood = float(average_precision_score(y_true, scores, pos_label=1))
    except Exception as e:
        logger.warning(f"AUPR-OOD calculation error: {e}")
        aupr_ood = float("nan")

    # 3. AUPR-In (In-distribution positive class, invert scores)
    try:
        aupr_in = float(average_precision_score(1 - y_true, -scores, pos_label=1))
    except Exception as e:
        logger.warning(f"AUPR-In calculation error: {e}")
        aupr_in = float("nan")

    # 4. FPR at 95% OOD Detection Rate (FPR95)
    fpr95 = compute_fpr95(y_true, scores)

    # 5. Operating point at frozen threshold
    if threshold is None:
        # Default operating threshold: 95th percentile of ID scores (5% false alarms on ID)
        threshold = float(np.percentile(id_s, 95.0))

    y_pred = (scores > threshold).astype(int)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # OOD Detection Rate
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # ID Retention Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0

    false_rejection_rate = fp / n_id if n_id > 0 else 0.0  # ID flagged as OOD
    false_acceptance_rate = fn / n_ood if n_ood > 0 else 0.0  # OOD accepted as ID

    conf_mat = [[tn, fp], [fn, tp]]

    return {
        "modality": modality,
        "method": method,
        "scenario": scenario,
        "n_id": n_id,
        "n_ood": n_ood,
        "auroc": round(auroc, 4),
        "aupr_ood": round(aupr_ood, 4),
        "aupr_in": round(aupr_in, 4),
        "fpr95": round(fpr95, 4),
        "threshold": round(float(threshold), 6),
        "sensitivity_ood": round(sensitivity, 4),
        "specificity_id": round(specificity, 4),
        "precision_ood": round(precision, 4),
        "f1_ood": round(f1, 4),
        "false_rejection_rate": round(false_rejection_rate, 4),
        "false_acceptance_rate": round(false_acceptance_rate, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "confusion_matrix": conf_mat,
        "id_mean_score": round(float(np.mean(id_s)), 6),
        "id_std_score": round(float(np.std(id_s)), 6),
        "ood_mean_score": round(float(np.mean(ood_s)), 6),
        "ood_std_score": round(float(np.std(ood_s)), 6),
    }


class OODEvaluator:
    """
    Evaluator class for OOD performance.
    Wraps evaluate_ood_metrics and compute_fpr95.
    """

    @staticmethod
    def evaluate(
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        threshold: Optional[float] = None,
        modality: str = "unknown",
        method: str = "unknown",
        scenario: str = "unknown",
    ) -> Dict[str, Any]:
        return evaluate_ood_metrics(
            id_scores=id_scores,
            ood_scores=ood_scores,
            threshold=threshold,
            modality=modality,
            method=method,
            scenario=scenario,
        )

    @staticmethod
    def compute_fpr95(y_true: np.ndarray, y_scores: np.ndarray) -> float:
        return compute_fpr95(y_true=y_true, y_scores=y_scores)
