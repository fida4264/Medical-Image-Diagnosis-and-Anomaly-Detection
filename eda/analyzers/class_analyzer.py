"""
eda/analyzers/class_analyzer.py
==================================
Analyzes disease class distribution, balance, and rarity.

Statistics Generated
---------------------
- Class frequency table
- Minority / majority class identification
- Rare disease detection (below configurable threshold)
- Class imbalance ratio (max/min and Gini coefficient)
- Per-modality class diversity
- Imbalance factor per modality

Classes
-------
ClassAnalyzer
    Computes class balance and imbalance statistics.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from eda.config import (
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_SPLIT,
    MODALITY_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


def _gini_coefficient(counts: np.ndarray) -> float:
    """Compute the Gini coefficient of a distribution (0 = perfect balance)."""
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    counts = np.sort(counts.astype(float))
    n = len(counts)
    index = np.arange(1, n + 1)
    gini = (2 * (index * counts).sum()) / (n * counts.sum()) - (n + 1) / n
    return round(float(np.clip(gini, 0.0, 1.0)), 4)


class ClassAnalyzer:
    """
    Analyzes disease class balance and rarity.

    Parameters
    ----------
    df : pd.DataFrame
        The master_metadata DataFrame (loaded by DatasetAnalyzer).
    rare_class_threshold : int
        Classes with fewer than this many images are marked "rare".
    minority_ratio_threshold : float
        Classes with count < minority_ratio_threshold × max_class_count
        are marked "minority".
    """

    def __init__(
        self,
        df: pd.DataFrame,
        rare_class_threshold: int = 100,
        minority_ratio_threshold: float = 0.1,
    ) -> None:
        self._df = df
        self._rare_thresh = rare_class_threshold
        self._minority_ratio = minority_ratio_threshold
        self._stats: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Compute class balance statistics.

        Returns
        -------
        dict
            Class balance statistics.
        """
        df = self._df
        logger.info("Running ClassAnalyzer ...")

        if COL_NORMALIZED_CLASS not in df.columns:
            logger.warning("normalized_class column not found — skipping class analysis.")
            return {}

        class_counts = df[COL_NORMALIZED_CLASS].value_counts()
        total = len(df)
        n_classes = len(class_counts)
        max_count = int(class_counts.iloc[0]) if n_classes > 0 else 0
        min_count = int(class_counts.iloc[-1]) if n_classes > 0 else 0

        # Imbalance ratio
        imbalance_ratio = round(max_count / min_count, 2) if min_count > 0 else float("inf")

        # Gini coefficient
        gini = _gini_coefficient(class_counts.values)

        # Minority / majority / rare
        minority_threshold = max_count * self._minority_ratio
        minority_classes: List[str] = []
        majority_classes: List[str] = []
        rare_classes: List[str] = []

        for cls, cnt in class_counts.items():
            if cnt < self._rare_thresh:
                rare_classes.append(cls)
            if cnt < minority_threshold:
                minority_classes.append(cls)
            else:
                majority_classes.append(cls)

        # Per-modality breakdown
        modality_class_stats: Dict[str, Any] = {}
        if COL_MODALITY in df.columns:
            for mod, grp in df.groupby(COL_MODALITY):
                mc = grp[COL_NORMALIZED_CLASS].value_counts()
                mc_max = int(mc.iloc[0]) if len(mc) > 0 else 0
                mc_min = int(mc.iloc[-1]) if len(mc) > 0 else 0
                modality_class_stats[mod] = {
                    "n_classes": len(mc),
                    "class_counts": mc.to_dict(),
                    "max_class_count": mc_max,
                    "min_class_count": mc_min,
                    "imbalance_ratio": round(mc_max / mc_min, 2) if mc_min > 0 else None,
                    "gini": _gini_coefficient(mc.values),
                    "display_name": MODALITY_DISPLAY_NAMES.get(mod, mod),
                }

        # Per-split class distribution
        split_class: Dict[str, Any] = {}
        if COL_SPLIT in df.columns:
            for split, grp in df.groupby(COL_SPLIT):
                split_class[split] = grp[COL_NORMALIZED_CLASS].value_counts().to_dict()

        # Top 10 and bottom 10 classes
        top10 = class_counts.head(10).to_dict()
        bottom10 = class_counts.tail(10).to_dict()

        # Balance assessment
        if gini < 0.1:
            balance_label = "Well Balanced"
        elif gini < 0.3:
            balance_label = "Slightly Imbalanced"
        elif gini < 0.5:
            balance_label = "Moderately Imbalanced"
        else:
            balance_label = "Severely Imbalanced"

        self._stats = {
            "total_classes": n_classes,
            "total_images": total,
            "class_counts": class_counts.to_dict(),
            "top10_classes": top10,
            "bottom10_classes": bottom10,
            "max_class": {"name": class_counts.index[0], "count": max_count} if n_classes > 0 else {},
            "min_class": {"name": class_counts.index[-1], "count": min_count} if n_classes > 0 else {},
            "imbalance_ratio": imbalance_ratio,
            "gini_coefficient": gini,
            "balance_label": balance_label,
            "minority_classes": minority_classes,
            "majority_classes": majority_classes,
            "rare_classes": rare_classes,
            "minority_class_count": len(minority_classes),
            "rare_class_count": len(rare_classes),
            "minority_threshold_count": int(minority_threshold),
            "rare_threshold_count": self._rare_thresh,
            "per_modality": modality_class_stats,
            "per_split_class_distribution": split_class,
        }

        logger.info(
            "ClassAnalyzer complete: %d classes | imbalance=%.1f× | gini=%.3f | %s",
            n_classes, imbalance_ratio, gini, balance_label,
        )
        return self._stats

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats
