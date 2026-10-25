"""
calibration/v2_phase7/comparator.py
=====================================
Baseline vs calibrated model comparison.

Loads baseline metrics from Phase 6 outputs and calibrated metrics from
Phase 7 temperature scaling. Produces a unified comparison table per modality.

Reported metrics:
  - Accuracy
  - Macro-F1
  - Weighted-F1
  - ROC-AUC
  - NLL (Negative Log-Likelihood)
  - Brier Score
  - ECE (Expected Calibration Error)
  - MCE (Maximum Calibration Error)

IMPORTANT:
  - Calibration primarily improves probability RELIABILITY, not accuracy.
  - A stable or unchanged Accuracy/F1 after calibration is expected and acceptable.
  - Calibration is NOT rejected if accuracy does not improve.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class CalibrationComparator:
    """
    Compares baseline and calibrated model metrics across all modalities.
    """

    def __init__(
        self,
        phase6_metrics_csv: Path,
        calibration_results: List[Dict[str, Any]],
    ):
        """
        Args:
            phase6_metrics_csv: Path to outputs/v2_phase6/per_modality_metrics.csv
            calibration_results: List of dicts from temperature_scaling module
        """
        self.phase6_csv = Path(phase6_metrics_csv)
        self.calibration_results = calibration_results
        self._baseline_df: Optional[pd.DataFrame] = None

    def _load_baseline(self) -> pd.DataFrame:
        if self._baseline_df is None:
            self._baseline_df = pd.read_csv(self.phase6_csv)
        return self._baseline_df

    def build_comparison_table(self) -> List[Dict[str, Any]]:
        """
        Builds the baseline vs calibrated comparison table.

        For each modality, pairs the baseline test metrics (from Phase 6) with
        the calibrated test metrics (from Phase 7 temperature scaling).

        Returns:
            List of per-modality comparison dicts
        """
        baseline_df = self._load_baseline()
        cal_lookup = {r["modality"]: r for r in self.calibration_results}

        rows = []
        for _, brow in baseline_df.iterrows():
            modality = str(brow.get("modality", ""))
            partition = str(brow.get("partition", ""))

            # Only compare test partitions (or CV mean for kfold modalities)
            if partition not in ("test", "CV_mean"):
                continue

            cal = cal_lookup.get(modality, {})

            row = {
                "modality": modality,
                "partition": partition,
                # Classification metrics (unchanged by temperature scaling)
                "accuracy_baseline": brow.get("accuracy"),
                "accuracy_calibrated": brow.get("accuracy"),  # unchanged
                "macro_f1_baseline": brow.get("macro_f1"),
                "macro_f1_calibrated": brow.get("macro_f1"),  # unchanged
                "weighted_f1_baseline": brow.get("weighted_f1"),
                "weighted_f1_calibrated": brow.get("weighted_f1"),
                "roc_auc_baseline": brow.get("roc_auc"),
                "roc_auc_calibrated": brow.get("roc_auc"),
                # Calibration metrics (change with temperature)
                "nll_baseline": cal.get("test_nll_before") or cal.get("val_nll_before"),
                "nll_calibrated": cal.get("test_nll_after") or cal.get("val_nll_after"),
                "brier_baseline": cal.get("test_brier_before") or cal.get("val_brier_before"),
                "brier_calibrated": cal.get("test_brier_after") or cal.get("val_brier_after"),
                "ece_baseline": cal.get("test_ece_before") or cal.get("val_ece_before"),
                "ece_calibrated": cal.get("test_ece_after") or cal.get("val_ece_after"),
                "mce_baseline": cal.get("test_mce_before") or cal.get("val_mce_before"),
                "mce_calibrated": cal.get("test_mce_after") or cal.get("val_mce_after"),
                "temperature": cal.get("temperature"),
                "note": (
                    "Temperature scaling improves probability reliability (ECE/NLL). "
                    "Accuracy/F1 unchanged — expected behavior."
                    if cal
                    else "No calibration data (retina uses per-label thresholds)"
                ),
            }
            rows.append(row)

        logger.info(f"Built comparison table with {len(rows)} modality rows")
        return rows

    def save(self, path: Path | str, rows: Optional[List[Dict[str, Any]]] = None) -> None:
        """Saves comparison table to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if rows is None:
            rows = self.build_comparison_table()
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(f"Calibration comparison table saved to {path}")

    def summary_stats(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes aggregate summary statistics across all modalities.

        Returns:
            Dict with mean ECE reduction, mean NLL reduction, etc.
        """
        ece_before = [r["ece_baseline"] for r in rows if r["ece_baseline"] is not None]
        ece_after = [r["ece_calibrated"] for r in rows if r["ece_calibrated"] is not None]
        nll_before = [r["nll_baseline"] for r in rows if r["nll_baseline"] is not None]
        nll_after = [r["nll_calibrated"] for r in rows if r["nll_calibrated"] is not None]

        import numpy as np
        return {
            "mean_ece_baseline": round(float(np.mean(ece_before)), 6) if ece_before else None,
            "mean_ece_calibrated": round(float(np.mean(ece_after)), 6) if ece_after else None,
            "mean_ece_reduction": round(
                float(np.mean(ece_before)) - float(np.mean(ece_after)), 6
            ) if ece_before and ece_after else None,
            "mean_nll_baseline": round(float(np.mean(nll_before)), 6) if nll_before else None,
            "mean_nll_calibrated": round(float(np.mean(nll_after)), 6) if nll_after else None,
            "mean_nll_reduction": round(
                float(np.mean(nll_before)) - float(np.mean(nll_after)), 6
            ) if nll_before and nll_after else None,
            "n_modalities_calibrated": len(ece_before),
        }
