"""
augmentation/reports/statistics.py
=====================================
Computes and saves augmentation_statistics.json.

Compares before/after class distributions, augmentation factors per
modality and class, and class balancing effectiveness.

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class AugmentationStatistics:
    """
    Compute before/after class distribution statistics.

    Usage
    -----
        stats = AugmentationStatistics()
        stats.load_original(preprocessing_metadata_df)
        stats.load_augmented(augmentation_metadata_df)
        stats.compute()
        stats.save(Path("augmentation_statistics.json"))
    """

    def __init__(self) -> None:
        self._original_df: Optional[pd.DataFrame] = None
        self._augmented_df: Optional[pd.DataFrame] = None
        self._statistics: Optional[Dict[str, Any]] = None

    def load_original(self, df: pd.DataFrame) -> None:
        """Load the original (pre-augmentation) metadata DataFrame."""
        self._original_df = df

    def load_augmented(self, df: pd.DataFrame) -> None:
        """Load the augmented metadata DataFrame."""
        self._augmented_df = df

    def compute(self) -> Dict[str, Any]:
        """
        Compute augmentation statistics.

        Returns
        -------
        dict
            Full statistics dictionary.
        """
        if self._original_df is None or self._augmented_df is None:
            raise RuntimeError(
                "Call load_original() and load_augmented() before compute()."
            )

        orig = self._original_df
        aug = self._augmented_df

        # --- Before ---
        before: Dict[str, Any] = {}
        for modality in orig["modality"].unique():
            mod_df = orig[orig["modality"] == modality]
            class_counts = {}
            if "normalized_class" in mod_df.columns:
                class_counts = mod_df["normalized_class"].value_counts().to_dict()
            before[modality] = {
                "total_images": len(mod_df),
                "per_class": class_counts,
                "num_classes": mod_df["normalized_class"].nunique()
                if "normalized_class" in mod_df.columns else 0,
            }

        # --- After (original + augmented combined) ---
        after: Dict[str, Any] = {}
        aug_success = aug[aug["status"] == "success"] if "status" in aug.columns else aug
        for modality in orig["modality"].unique():
            orig_count = len(orig[orig["modality"] == modality])
            aug_mod = aug_success[aug_success["modality"] == modality]
            aug_count = len(aug_mod)
            total = orig_count + aug_count

            class_counts_after = {}
            if "normalized_class" in orig.columns:
                orig_cls = orig[orig["modality"] == modality]["normalized_class"].value_counts().to_dict()
                aug_cls = aug_mod["normalized_class"].value_counts().to_dict() if "normalized_class" in aug_mod.columns else {}
                for cls in set(list(orig_cls.keys()) + list(aug_cls.keys())):
                    class_counts_after[cls] = orig_cls.get(cls, 0) + aug_cls.get(cls, 0)

            after[modality] = {
                "total_images": total,
                "original_images": orig_count,
                "augmented_images": aug_count,
                "augmentation_factor": round(aug_count / orig_count, 2) if orig_count > 0 else 0.0,
                "per_class": class_counts_after,
            }

        # --- Imbalance metrics ---
        imbalance: Dict[str, Any] = {}
        for modality in before:
            per_cls_before = before[modality].get("per_class", {})
            per_cls_after = after.get(modality, {}).get("per_class", {})
            if per_cls_before:
                max_b = max(per_cls_before.values())
                min_b = min(per_cls_before.values())
                max_a = max(per_cls_after.values()) if per_cls_after else 0
                min_a = min(per_cls_after.values()) if per_cls_after else 0
                imbalance[modality] = {
                    "before_ratio": round(min_b / max_b, 3) if max_b > 0 else 1.0,
                    "after_ratio": round(min_a / max_a, 3) if max_a > 0 else 1.0,
                    "improved": (min_a / max_a > min_b / max_b) if max_b > 0 and max_a > 0 else False,
                }

        # --- Overall summary ---
        total_orig = len(orig)
        total_aug = len(aug_success)

        self._statistics = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_original_images": total_orig,
            "total_augmented_images": total_aug,
            "total_combined_images": total_orig + total_aug,
            "overall_augmentation_factor": round(total_aug / total_orig, 2) if total_orig > 0 else 0.0,
            "before_augmentation": before,
            "after_augmentation": after,
            "class_imbalance": imbalance,
        }

        return self._statistics

    def save(self, output_path: Path) -> Path:
        """
        Write augmentation_statistics.json.

        Raises
        ------
        RuntimeError
            If compute() has not been called yet.
        """
        if self._statistics is None:
            raise RuntimeError("Call compute() before save().")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(self._statistics, fh, indent=2, default=str)
        logger.info("augmentation_statistics.json saved → %s", output_path)
        return output_path

    @property
    def statistics(self) -> Optional[Dict[str, Any]]:
        """Return computed statistics or None if compute() not called."""
        return self._statistics
