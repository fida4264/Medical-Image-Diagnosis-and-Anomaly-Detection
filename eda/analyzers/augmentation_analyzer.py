"""
eda/analyzers/augmentation_analyzer.py
=========================================
Analyzes augmentation outcomes from Module 4's augmentation_metadata.csv
and augmentation_report.json.

Statistics Generated
---------------------
- Images before / after augmentation
- Augmentation factor per modality
- Class balance improvement (before vs after)
- Transformation frequency
- Success / error rates
- Per-modality augmentation stats

Classes
-------
AugmentationAnalyzer
    Reads augmentation metadata and computes augmentation analytics.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.config import (
    AUGMENTATION_METADATA_CSV,
    AUGMENTATION_REPORT_JSON,
    AUGMENTATION_STATISTICS_JSON,
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_AUG_STATUS,
    COL_AUG_OPS,
    COL_AUG_INDEX,
)

logger = logging.getLogger(__name__)


class AugmentationAnalyzer:
    """
    Analyzes Module 4 augmentation outcomes.

    Parameters
    ----------
    metadata_csv : Path, optional
        Path to augmentation_metadata.csv.
    report_json : Path, optional
        Path to augmentation_report.json.
    statistics_json : Path, optional
        Path to augmentation_statistics.json.
    master_df : pd.DataFrame, optional
        master_metadata DataFrame for computing before/after class balance.
    """

    def __init__(
        self,
        metadata_csv: Optional[Path] = None,
        report_json: Optional[Path] = None,
        statistics_json: Optional[Path] = None,
        master_df: Optional[pd.DataFrame] = None,
    ) -> None:
        self._csv = Path(metadata_csv) if metadata_csv else AUGMENTATION_METADATA_CSV
        self._report_json = Path(report_json) if report_json else AUGMENTATION_REPORT_JSON
        self._stats_json = Path(statistics_json) if statistics_json else AUGMENTATION_STATISTICS_JSON
        self._master_df = master_df
        self._df: Optional[pd.DataFrame] = None
        self._report: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}

    def load(self) -> "AugmentationAnalyzer":
        """Load augmentation metadata files."""
        if self._csv.exists():
            self._df = pd.read_csv(self._csv, encoding="utf-8", low_memory=False)
            logger.info("Loaded augmentation_metadata.csv: %d rows", len(self._df))
        else:
            logger.warning("augmentation_metadata.csv not found at %s", self._csv)

        for path, attr in [(self._report_json, "_report")]:
            if path.exists():
                with open(path, "r", encoding="utf-8") as fh:
                    setattr(self, attr, json.load(fh))
        return self

    def analyze(self) -> Dict[str, Any]:
        """
        Compute augmentation statistics.

        Returns
        -------
        dict
            Augmentation analytics dictionary.
        """
        if self._df is None:
            self.load()

        if self._df is None or self._df.empty:
            logger.warning("No augmentation data available.")
            return {"available": False}

        logger.info("Running AugmentationAnalyzer ...")
        df = self._df

        # --- Status counts ---
        status_counts: Dict[str, int] = {}
        if COL_AUG_STATUS in df.columns:
            status_counts = df[COL_AUG_STATUS].value_counts().to_dict()

        total = len(df)
        success = status_counts.get("success", 0)
        errors = sum(v for k, v in status_counts.items() if k.startswith("error"))

        # --- Augmentation factor ---
        aug_factor = self._report.get("augmentation_factor", 0.0)
        total_original = self._report.get("total_original_images", 0)

        # --- Per-modality stats ---
        modality_stats: Dict[str, Any] = {}
        if COL_MODALITY in df.columns:
            for mod, grp in df.groupby(COL_MODALITY):
                sc = grp[COL_AUG_STATUS].value_counts().to_dict() if COL_AUG_STATUS in grp.columns else {}
                n_success = sc.get("success", 0)
                n_total = len(grp)
                # Classes represented
                n_classes = grp[COL_NORMALIZED_CLASS].nunique() if COL_NORMALIZED_CLASS in grp.columns else 0
                # Average augmentations per original image
                if COL_AUG_INDEX in grp.columns:
                    n_unique_originals = grp["original_image_id"].nunique() if "original_image_id" in grp.columns else 0
                    avg_factor = round(n_success / n_unique_originals, 2) if n_unique_originals > 0 else 0.0
                else:
                    avg_factor = 0.0

                modality_stats[mod] = {
                    "total_augmented": n_success,
                    "total_records": n_total,
                    "success_rate_pct": round(n_success / n_total * 100, 2) if n_total > 0 else 0.0,
                    "n_classes": n_classes,
                    "avg_aug_factor": avg_factor,
                }

        # --- Transformation frequency ---
        ops_freq: Dict[str, int] = {}
        if COL_AUG_OPS in df.columns:
            for ops_str in df[COL_AUG_OPS].dropna():
                for op in str(ops_str).split("|"):
                    op = op.strip()
                    if op:
                        ops_freq[op] = ops_freq.get(op, 0) + 1
            ops_freq = dict(sorted(ops_freq.items(), key=lambda x: x[1], reverse=True))

        # --- Class distribution before / after ---
        class_balance = self._compute_class_balance_improvement(df)

        # --- Per-class augmented count ---
        class_counts_after: Dict[str, int] = {}
        if COL_NORMALIZED_CLASS in df.columns and COL_AUG_STATUS in df.columns:
            class_counts_after = (
                df[df[COL_AUG_STATUS] == "success"][COL_NORMALIZED_CLASS]
                .value_counts()
                .to_dict()
            )

        self._stats = {
            "available": True,
            "total_augmented_records": total,
            "total_successful_augmentations": success,
            "total_errors": errors,
            "success_rate_pct": round(success / total * 100, 2) if total > 0 else 0.0,
            "total_original_images": total_original,
            "augmentation_factor": aug_factor,
            "status_counts": status_counts,
            "per_modality": modality_stats,
            "transformation_frequency": ops_freq,
            "class_balance_improvement": class_balance,
            "class_counts_after_augmentation": class_counts_after,
            "report_summary": self._report,
        }

        logger.info(
            "AugmentationAnalyzer complete: %d records | factor=%.1f×",
            total, aug_factor,
        )
        return self._stats

    def _compute_class_balance_improvement(
        self, aug_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Compare class distribution before (master_df) and after augmentation.
        """
        result: Dict[str, Any] = {"available": False}

        if self._master_df is None or COL_NORMALIZED_CLASS not in self._master_df.columns:
            return result

        before = self._master_df[COL_NORMALIZED_CLASS].value_counts()
        if COL_NORMALIZED_CLASS not in aug_df.columns or COL_AUG_STATUS not in aug_df.columns:
            return result

        aug_only = aug_df[aug_df[COL_AUG_STATUS] == "success"][COL_NORMALIZED_CLASS].value_counts()

        # Combined (original + augmented)
        combined = before.add(aug_only, fill_value=0).astype(int)

        # Gini before and after
        def _gini(counts: np.ndarray) -> float:
            if len(counts) == 0 or counts.sum() == 0:
                return 0.0
            c = np.sort(counts.astype(float))
            n = len(c)
            idx = np.arange(1, n + 1)
            return float(np.clip((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n, 0, 1))

        gini_before = round(_gini(before.values), 4)
        gini_after = round(_gini(combined.values), 4)

        return {
            "available": True,
            "gini_before": gini_before,
            "gini_after": gini_after,
            "improvement": round(gini_before - gini_after, 4),
            "class_counts_before": before.to_dict(),
            "class_counts_after": combined.to_dict(),
            "min_class_before": int(before.min()) if not before.empty else 0,
            "min_class_after": int(combined.min()) if not combined.empty else 0,
            "max_class_before": int(before.max()) if not before.empty else 0,
            "max_class_after": int(combined.max()) if not combined.empty else 0,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._df
