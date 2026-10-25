"""
calibration/v2_phase7/cost_analysis.py
=======================================
Cost-sensitive decision threshold analysis.

For each clinically prioritized class, sweeps decision thresholds on
VALIDATION data and records the FP/FN trade-off at each threshold.

This is RESEARCH DECISION ANALYSIS only.
No clinical claims, recommendations, or diagnostic guidance are made.

Prioritized classes (per Phase 6 Phase 7 spec):
  1. Brain CT       — hemorrhage
  2. Skin Lesion    — melanoma
  3. Ultrasound     — malignant
  4. Chest X-ray    — lung_opacity
  5. Retina         — hypertension, amd, diabetic_retinopathy

Output:
  outputs/v2_phase7/cost_sensitive/cost_sensitive_analysis.csv
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

ANALYSIS_TARGETS = {
    "brain_ct": {
        "target_class": "hemorrhage",
        "target_idx": 1,
    },
    "skin": {
        "target_class": "melanoma",
        "target_idx": 1,
    },
    "ultrasound": {
        "target_class": "malignant",
        "target_idx": 1,
    },
    "chest_xray": {
        "target_class": "lung_opacity",
        "target_idx": 1,
    },
    "retina": {
        "target_class": "hypertension",
        "target_idx": 5,
    },
    "retina_amd": {
        "target_class": "amd",
        "target_idx": 4,
    },
    "retina_dr": {
        "target_class": "diabetic_retinopathy",
        "target_idx": 1,
    },
}


def _threshold_metrics(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Computes binary classification metrics at a single threshold.

    Args:
        y_true_binary: Binary ground truth (0/1), shape (N,)
        y_score: Predicted probability for the positive class, shape (N,)
        threshold: Decision threshold

    Returns:
        Dict with TP, FP, TN, FN, sensitivity, specificity, precision, recall, F1
    """
    pred = (y_score >= threshold).astype(int)
    tp = int(((pred == 1) & (y_true_binary == 1)).sum())
    fp = int(((pred == 1) & (y_true_binary == 0)).sum())
    tn = int(((pred == 0) & (y_true_binary == 0)).sum())
    fn = int(((pred == 0) & (y_true_binary == 1)).sum())

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0
        else 0.0
    )
    ppv = precision
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "threshold": round(threshold, 3),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(ppv, 4),
        "recall": round(sensitivity, 4),
        "npv": round(npv, 4),
        "f1": round(f1, 4),
    }


class CostSensitiveAnalyzer:
    """
    Analyzes FP/FN trade-offs across threshold values for prioritized classes.
    Operates on validation data only.
    """

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def analyze_modality(
        self,
        modality_key: str,
        target_class: str,
        target_idx: int,
        val_probs: np.ndarray,
        val_labels: np.ndarray,
        task: str = "multiclass",
    ) -> List[Dict[str, Any]]:
        """
        Sweeps thresholds and records cost-sensitive metrics.

        Args:
            modality_key: Modality identifier (e.g. 'brain_ct')
            target_class: Name of the positive class (e.g. 'hemorrhage')
            target_idx: Column index for target class probability
            val_probs: Probability array (N, C) — softmax for multiclass,
                       sigmoid for multilabel
            val_labels: Label array — integer (N,) for multiclass,
                        binary matrix (N, C) for multilabel
            task: 'multiclass', 'binary', or 'multilabel'

        Returns:
            List of threshold-level metric dicts
        """
        # Extract target class probability and binary true label
        y_score = val_probs[:, target_idx]

        if task == "multilabel":
            y_true_binary = val_labels[:, target_idx].astype(int)
        else:
            # Multiclass or binary: binarize against target class
            if val_labels.ndim == 1:
                y_true_binary = (val_labels == target_idx).astype(int)
            else:
                y_true_binary = val_labels[:, target_idx].astype(int)

        support = int(y_true_binary.sum())
        rows = []

        for t in np.arange(0.10, 0.91, 0.05):
            metrics = _threshold_metrics(y_true_binary, y_score, float(t))
            row = {
                "modality": modality_key,
                "target_class": target_class,
                "support": support,
                "total_samples": len(y_true_binary),
                "analysis_type": "research_decision_analysis",
                **metrics,
            }
            rows.append(row)

        logger.info(
            f"[cost_analysis/{modality_key}/{target_class}] "
            f"support={support}/{len(y_true_binary)} | "
            f"Analyzed {len(rows)} threshold points"
        )
        self.results.extend(rows)
        return rows

    def save(self, path: Path | str) -> None:
        """Saves all cost-sensitive analysis results to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not self.results:
            logger.warning("No cost-sensitive analysis results to save.")
            return
        fieldnames = list(self.results[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)
        logger.info(f"Cost-sensitive analysis saved to {path} ({len(self.results)} rows)")
