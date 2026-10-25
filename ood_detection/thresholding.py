"""
ood_detection/thresholding.py
=============================
Operating threshold selection for OOD detection.
Selects thresholds strictly using VALIDATION data only.
Test data is NEVER used for threshold selection.

Supported strategies:
1. 'tpr95': 95% True Positive Rate on in-distribution data (5% false alarm rate on ID).
   Selects threshold at 95th percentile of in-distribution validation scores.
   Requires only ID validation data (zero OOD assumption).
2. 'youden': Maximizes Youden's J index (Sensitivity + Specificity - 1) on validation ID vs validation OOD.
3. 'f1_optimal': Maximizes F1 score for OOD detection on validation ID vs validation OOD.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class OODThresholdSelector:
    """
    Selects and freezes operating decision thresholds for OOD detectors.
    Operates strictly on validation data.
    """

    def __init__(self, strategy: str = "tpr95"):
        self.strategy = strategy.lower()
        self.threshold: Optional[float] = None
        self.metadata: Dict[str, Any] = {}
        self.is_frozen: bool = False

    def fit_threshold(
        self,
        val_id_scores: np.ndarray,
        val_ood_scores: Optional[np.ndarray] = None,
        strategy: Optional[str] = None,
    ) -> float:
        """
        Determines the decision threshold using validation scores only.

        Score orientation: HIGHER = MORE OOD.
        Decision rule: If score > threshold -> PREDICT OOD.
                       If score <= threshold -> PREDICT ID.

        Args:
            val_id_scores: OOD scores on In-Distribution validation data, shape (N_val_id,).
            val_ood_scores: Optional OOD scores on Validation OOD data, shape (N_val_ood,).
            strategy: Optional override for strategy ('tpr95', 'youden', 'f1_optimal').

        Returns:
            Selected threshold value (float).
        """
        strat = strategy.lower() if strategy else self.strategy
        val_id = np.asarray(val_id_scores, dtype=np.float64)

        if strat == "tpr95":
            # 95% of in-distribution samples should have score <= threshold (accepted as ID)
            # Exactly 5% false rejection rate on ID validation data
            thresh = float(np.percentile(val_id, 95.0))
            self.metadata = {
                "strategy": "tpr95",
                "target_id_tpr": 0.95,
                "val_id_samples": len(val_id),
                "val_ood_samples": len(val_ood_scores) if val_ood_scores is not None else 0,
            }

        elif strat in ("youden", "f1_optimal"):
            if val_ood_scores is None or len(val_ood_scores) == 0:
                logger.warning(
                    f"Strategy '{strat}' requested but no validation OOD scores provided. "
                    "Falling back to 'tpr95'."
                )
                return self.fit_threshold(val_id, strategy="tpr95")

            val_ood = np.asarray(val_ood_scores, dtype=np.float64)
            # Candidate threshold search grid spanning both distributions
            all_scores = np.concatenate([val_id, val_ood])
            candidates = np.percentile(all_scores, np.linspace(1, 99, 100))

            best_score = -1.0
            best_thresh = float(np.median(candidates))

            for t in candidates:
                # OOD is positive class: predicted_ood = (score > t)
                tp = int(np.sum(val_ood > t))  # OOD correctly identified
                fn = int(np.sum(val_ood <= t)) # OOD missed
                fp = int(np.sum(val_id > t))   # ID incorrectly flagged as OOD
                tn = int(np.sum(val_id <= t))  # ID correctly accepted

                sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                f1 = 2 * prec * sens / (prec + sens) if (prec + sens) > 0 else 0.0

                if strat == "youden":
                    j_stat = sens + spec - 1.0
                    if j_stat > best_score:
                        best_score = j_stat
                        best_thresh = float(t)
                elif strat == "f1_optimal":
                    if f1 > best_score:
                        best_score = f1
                        best_thresh = float(t)

            thresh = best_thresh
            self.metadata = {
                "strategy": strat,
                "best_validation_metric": round(best_score, 4),
                "val_id_samples": len(val_id),
                "val_ood_samples": len(val_ood),
            }

        else:
            raise ValueError(f"Unknown threshold selection strategy: '{strat}'")

        self.threshold = float(thresh)
        self.is_frozen = True
        logger.info(f"OOD threshold frozen at {self.threshold:.6f} via '{strat}' on validation data.")
        return self.threshold

    def predict(self, scores: np.ndarray) -> np.ndarray:
        """
        Classifies inputs as ID (0) or OOD (1).

        Args:
            scores: OOD scores where higher = more OOD.

        Returns:
            Binary numpy array where 1 = OOD, 0 = In-Distribution.
        """
        if not self.is_frozen or self.threshold is None:
            raise RuntimeError("Threshold must be fitted and frozen before prediction.")
        scores_np = np.asarray(scores, dtype=np.float64)
        return (scores_np > self.threshold).astype(int)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "threshold": self.threshold,
            "is_frozen": self.is_frozen,
            "metadata": self.metadata,
        }
