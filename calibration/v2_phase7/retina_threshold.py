"""
calibration/v2_phase7/retina_threshold.py
==========================================
Per-label threshold optimization for Retina multi-label classification (ODIR-5K).

Retina uses 8 independent sigmoid outputs (NOT softmax multiclass).
Default threshold = 0.50 for all labels.

This module sweeps candidate thresholds on VALIDATION data only
and selects the threshold that maximizes per-label F1 score.

Special attention is given to low-support labels:
  - hypertension (support ~24, F1 baseline = 0.2667)
  - amd          (support ~37, F1 baseline = 0.3673)
  - pathological_myopia (support ~23, F1 baseline = 0.6923)

After thresholds are frozen (fit on validation), the test set is
evaluated EXACTLY ONCE. Test labels are never seen during threshold search.

Outputs:
  outputs/v2_phase7/retina_thresholds/retina_thresholds.json
  outputs/v2_phase7/retina_thresholds/retina_threshold_analysis.csv
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

RETINA_LABELS = [
    "normal",
    "diabetic_retinopathy",
    "glaucoma",
    "cataract",
    "amd",
    "hypertension",
    "pathological_myopia",
    "other_abnormality",
]

LOW_SUPPORT_LABELS = {"amd", "hypertension", "pathological_myopia"}

# Threshold candidate sweep range
THRESHOLD_MIN = 0.05
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Safe element-wise sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88, 88)))


def _precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Tuple[float, float, float]:
    """Binary precision, recall, F1 for a single label."""
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return precision, recall, f1


def _auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Manual AUROC via trapezoidal integration."""
    from sklearn.metrics import roc_auc_score
    try:
        if y_true.sum() == 0 or y_true.sum() == len(y_true):
            return float("nan")
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return float("nan")


def _youden_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Youden's J index threshold: argmax(sensitivity + specificity - 1).
    Useful for low-support labels where F1 optimization might collapse.
    """
    thresholds = np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP, THRESHOLD_STEP)
    best_j = -1.0
    best_t = 0.50
    for t in thresholds:
        pred = (y_score >= t).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        tn = int(((pred == 0) & (y_true == 0)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        j = sensitivity + specificity - 1.0
        if j > best_j:
            best_j = j
            best_t = float(t)
    return best_t


class RetinaThresholdOptimizer:
    """
    Optimizes per-label decision thresholds for Retina multi-label classification.

    Fitting is performed exclusively on validation sigmoid probabilities.
    Test set is evaluated once after thresholds are frozen.
    """

    def __init__(self):
        self.thresholds: Dict[str, float] = {label: 0.50 for label in RETINA_LABELS}
        self._fitted = False

    def fit(
        self,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
    ) -> Dict[str, float]:
        """
        Optimizes thresholds per label using validation data.

        Args:
            val_probs: Sigmoid probabilities, shape (N_val, 8)
            val_labels: Binary label matrix, shape (N_val, 8)

        Returns:
            Dict mapping label name → optimal threshold
        """
        assert val_probs.shape[1] == 8, "Expected 8 Retina labels"
        assert val_labels.shape[1] == 8, "Expected 8 Retina labels"

        thresholds = {}
        for i, label in enumerate(RETINA_LABELS):
            scores = val_probs[:, i]
            truths = val_labels[:, i].astype(int)
            support = int(truths.sum())

            # Sweep F1 across candidate thresholds
            best_f1 = -1.0
            best_t = 0.50
            for t in np.arange(THRESHOLD_MIN, THRESHOLD_MAX + THRESHOLD_STEP / 2, THRESHOLD_STEP):
                pred = (scores >= t).astype(int)
                _, _, f1 = _precision_recall_f1(truths, pred)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)

            # For low-support labels, also compute Youden threshold
            if label in LOW_SUPPORT_LABELS:
                youden_t = _youden_threshold(truths, scores)
                youden_pred = (scores >= youden_t).astype(int)
                _, _, youden_f1 = _precision_recall_f1(truths, youden_pred)
                logger.info(
                    f"[retina/{label}] support={support} | "
                    f"F1-optimal T={best_t:.2f} (F1={best_f1:.4f}) | "
                    f"Youden T={youden_t:.2f} (F1={youden_f1:.4f})"
                )
                # Choose whichever gives better F1
                if youden_f1 > best_f1:
                    best_t = youden_t
                    best_f1 = youden_f1
                    logger.info(f"[retina/{label}] Using Youden threshold {best_t:.2f}")

            thresholds[label] = round(best_t, 4)
            logger.info(f"[retina/{label}] support={support} | Optimal T={best_t:.2f} | Val F1={best_f1:.4f}")

        self.thresholds = thresholds
        self._fitted = True
        return thresholds

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """
        Applies per-label thresholds to produce binary predictions.

        Args:
            probs: Sigmoid probabilities, shape (N, 8)

        Returns:
            Binary prediction matrix, shape (N, 8)
        """
        assert self._fitted, "Call fit() before predict()"
        preds = np.zeros_like(probs, dtype=int)
        for i, label in enumerate(RETINA_LABELS):
            preds[:, i] = (probs[:, i] >= self.thresholds[label]).astype(int)
        return preds

    def evaluate(
        self,
        probs: np.ndarray,
        labels: np.ndarray,
        use_optimized: bool = True,
        split_name: str = "eval",
    ) -> List[Dict[str, Any]]:
        """
        Evaluates per-label metrics with default (0.50) and optimized thresholds.

        Args:
            probs: Sigmoid probabilities (N, 8)
            labels: Binary label matrix (N, 8)
            use_optimized: If True, uses fitted thresholds; else uses 0.50
            split_name: Identifier for logging

        Returns:
            List of per-label metric dicts
        """
        rows = []
        for i, label in enumerate(RETINA_LABELS):
            scores = probs[:, i]
            truths = labels[:, i].astype(int)
            support = int(truths.sum())

            # Default threshold
            pred_default = (scores >= 0.50).astype(int)
            prec_def, rec_def, f1_def = _precision_recall_f1(truths, pred_default)
            auroc = _auroc(truths, scores)

            row = {
                "label": label,
                "split": split_name,
                "support": support,
                "default_threshold": 0.50,
                "default_precision": round(prec_def, 4),
                "default_recall": round(rec_def, 4),
                "default_f1": round(f1_def, 4),
                "auroc": round(auroc, 4) if not np.isnan(auroc) else None,
            }

            # Optimized threshold
            if use_optimized and self._fitted:
                t_opt = self.thresholds[label]
                pred_opt = (scores >= t_opt).astype(int)
                prec_opt, rec_opt, f1_opt = _precision_recall_f1(truths, pred_opt)
                row.update({
                    "optimized_threshold": t_opt,
                    "optimized_precision": round(prec_opt, 4),
                    "optimized_recall": round(rec_opt, 4),
                    "optimized_f1": round(f1_opt, 4),
                    "f1_improvement": round(f1_opt - f1_def, 4),
                    "threshold_method": "youden_or_f1_max_val",
                    "low_support_label": label in LOW_SUPPORT_LABELS,
                })

            rows.append(row)
        return rows

    def save(self, path: Path | str) -> None:
        """Saves thresholds to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "retina_thresholds": self.thresholds,
            "default_threshold": 0.50,
            "method": "max_f1_validation_sweep",
            "threshold_range": f"{THRESHOLD_MIN}–{THRESHOLD_MAX}",
            "threshold_step": THRESHOLD_STEP,
            "low_support_labels": sorted(LOW_SUPPORT_LABELS),
            "fitted_on": "validation_split_only",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Retina thresholds saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "RetinaThresholdOptimizer":
        """Loads thresholds from JSON."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        opt = cls()
        opt.thresholds = data["retina_thresholds"]
        opt._fitted = True
        return opt

    def save_analysis_csv(
        self, rows: List[Dict[str, Any]], path: Path | str
    ) -> None:
        """Saves per-label threshold analysis to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Retina threshold analysis saved to {path}")
