"""
anomaly_detection/evaluator.py
==============================
Evaluation metrics for within-distribution anomaly detection.
Measures discrimination between healthy/normal reference samples and pathological/disease samples.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

logger = logging.getLogger(__name__)


class AnomalyEvaluator:
    """
    Evaluates anomaly detection performance (Normal vs Pathological within-distribution).
    Positive class (1) = Pathological / Anomalous
    Negative class (0) = Normal / Healthy Reference
    """

    @staticmethod
    def evaluate(
        normal_scores: np.ndarray,
        pathological_scores: np.ndarray,
        modality: str,
        reference_name: str = "normal",
    ) -> Dict[str, Any]:
        """
        Evaluates AUROC and separation metrics.

        Args:
            normal_scores: Anomaly scores for confirmed normal/healthy cases.
            pathological_scores: Anomaly scores for confirmed pathological/disease cases.
            modality: Modality name.
            reference_name: Name of reference normal state.

        Returns:
            Dict containing AUROC, AUPR, and operating points.
        """
        n_norm = len(normal_scores)
        n_path = len(pathological_scores)

        if n_norm == 0 or n_path == 0:
            return {
                "modality": modality,
                "reference_class": reference_name,
                "n_normal": n_norm,
                "n_pathological": n_path,
                "status": "unavailable_insufficient_samples",
            }

        y_true = np.concatenate([np.zeros(n_norm, dtype=int), np.ones(n_path, dtype=int)])
        scores = np.concatenate([normal_scores, pathological_scores])

        try:
            auroc = float(roc_auc_score(y_true, scores))
        except Exception:
            auroc = float("nan")

        try:
            aupr = float(average_precision_score(y_true, scores, pos_label=1))
        except Exception as e:
            aupr = float("nan")

        # Threshold at 90% normal specificity (accepts 90% of normal cases)
        thresh_90 = float(np.percentile(normal_scores, 90.0))
        sens_at_90spec = float(np.mean(pathological_scores > thresh_90))

        return {
            "modality": modality,
            "reference_class": reference_name,
            "n_normal": n_norm,
            "n_pathological": n_path,
            "anomaly_auroc": round(auroc, 4),
            "anomaly_aupr": round(aupr, 4),
            "threshold_90_spec": round(thresh_90, 4),
            "sensitivity_at_90_spec": round(sens_at_90spec, 4),
            "normal_mean_score": round(float(np.mean(normal_scores)), 4),
            "pathological_mean_score": round(float(np.mean(pathological_scores)), 4),
        }
