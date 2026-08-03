"""
eda/analyzers/modality_analyzer.py
=====================================
Per-modality breakdown: image counts, class diversity, resolution, file size.

Classes
-------
ModalityAnalyzer
    Computes per-modality statistics from master_metadata.csv.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from eda.config import (
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_WIDTH,
    COL_HEIGHT,
    COL_FILE_SIZE,
    COL_CHANNELS,
    COL_EXTENSION,
    COL_SPLIT,
    MODALITY_DISPLAY_NAMES,
    GRAYSCALE_MODALITIES,
    RGB_MODALITIES,
)

logger = logging.getLogger(__name__)


class ModalityAnalyzer:
    """
    Computes per-modality statistics.

    Parameters
    ----------
    df : pd.DataFrame
        The master_metadata DataFrame.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._stats: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Compute per-modality statistics.

        Returns
        -------
        dict
            Per-modality statistics and cross-modality summary.
        """
        df = self._df
        logger.info("Running ModalityAnalyzer ...")

        if COL_MODALITY not in df.columns:
            logger.warning("modality column not found — skipping modality analysis.")
            return {}

        per_modality: Dict[str, Any] = {}

        for mod, grp in df.groupby(COL_MODALITY):
            n = len(grp)
            display = MODALITY_DISPLAY_NAMES.get(mod, mod.replace("_", " ").title())

            # Class diversity
            n_classes = (
                grp[COL_NORMALIZED_CLASS].nunique()
                if COL_NORMALIZED_CLASS in grp.columns else 0
            )
            class_distribution = (
                grp[COL_NORMALIZED_CLASS].value_counts().to_dict()
                if COL_NORMALIZED_CLASS in grp.columns else {}
            )

            # Resolution
            res: Dict[str, Any] = {}
            if COL_WIDTH in grp.columns and COL_HEIGHT in grp.columns:
                valid = grp[(grp[COL_WIDTH] > 0) & (grp[COL_HEIGHT] > 0)]
                if not valid.empty:
                    res = {
                        "width_mean": round(float(valid[COL_WIDTH].mean()), 1),
                        "width_std": round(float(valid[COL_WIDTH].std()), 1),
                        "width_min": int(valid[COL_WIDTH].min()),
                        "width_max": int(valid[COL_WIDTH].max()),
                        "height_mean": round(float(valid[COL_HEIGHT].mean()), 1),
                        "height_std": round(float(valid[COL_HEIGHT].std()), 1),
                        "height_min": int(valid[COL_HEIGHT].min()),
                        "height_max": int(valid[COL_HEIGHT].max()),
                        "aspect_ratio_mean": round(
                            float((valid[COL_WIDTH] / valid[COL_HEIGHT]).mean()), 3
                        ),
                    }

            # File size
            size: Dict[str, Any] = {}
            if COL_FILE_SIZE in grp.columns:
                fs = grp[COL_FILE_SIZE].dropna()
                if not fs.empty:
                    size = {
                        "mean_kb": round(float(fs.mean()) / 1024, 2),
                        "median_kb": round(float(fs.median()) / 1024, 2),
                        "total_mb": round(float(fs.sum()) / (1024 ** 2), 2),
                    }

            # Channel type
            is_grayscale = mod in GRAYSCALE_MODALITIES
            colour_type = "Grayscale (3-ch replicated)" if is_grayscale else "RGB"

            # Split distribution
            split_dist: Dict[str, int] = {}
            if COL_SPLIT in grp.columns:
                split_dist = grp[COL_SPLIT].fillna("unknown").value_counts().to_dict()

            # Format distribution
            fmt_dist: Dict[str, int] = {}
            if COL_EXTENSION in grp.columns:
                fmt_dist = grp[COL_EXTENSION].str.lower().value_counts().to_dict()

            per_modality[mod] = {
                "display_name": display,
                "total_images": n,
                "n_classes": n_classes,
                "class_distribution": class_distribution,
                "colour_type": colour_type,
                "is_grayscale": is_grayscale,
                "resolution": res,
                "file_size": size,
                "split_distribution": split_dist,
                "format_distribution": fmt_dist,
            }

        # Cross-modality aggregates
        modality_counts = {m: v["total_images"] for m, v in per_modality.items()}
        largest_modality = max(modality_counts, key=modality_counts.get) if modality_counts else ""
        smallest_modality = min(modality_counts, key=modality_counts.get) if modality_counts else ""

        self._stats = {
            "per_modality": per_modality,
            "total_modalities": len(per_modality),
            "largest_modality": largest_modality,
            "smallest_modality": smallest_modality,
            "grayscale_modalities": sorted(GRAYSCALE_MODALITIES),
            "rgb_modalities": sorted(RGB_MODALITIES),
        }

        logger.info("ModalityAnalyzer complete: %d modalities", len(per_modality))
        return self._stats

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats
