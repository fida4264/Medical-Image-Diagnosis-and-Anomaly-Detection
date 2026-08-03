"""
eda/analyzers/quality_analyzer.py
====================================
Analyzes image quality from Module 3's quality_report.json and
computes the Dataset Quality Score.

Quality Score Components (0–100 total)
---------------------------------------
1. Class Balance Score   (0–25): based on class Gini coefficient
2. Image Quality Score   (0–25): 1 − overall flag rate
3. Corruption Score      (0–25): 1 − corruption/error rate
4. Resolution Score      (0–25): % of images within ±20% of median dimension

Classes
-------
QualityAnalyzer
    Reads quality_report.json and computes quality metrics + composite score.

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
    QUALITY_REPORT_JSON,
    COL_WIDTH,
    COL_HEIGHT,
    COL_MODALITY,
)

logger = logging.getLogger(__name__)

# Quality score weight constants
WEIGHT_CLASS_BALANCE    = 25
WEIGHT_IMAGE_QUALITY    = 25
WEIGHT_CORRUPTION       = 25
WEIGHT_RESOLUTION       = 25


class QualityAnalyzer:
    """
    Reads quality_report.json from Module 3 and computes quality stats.

    Parameters
    ----------
    quality_report_path : Path, optional
        Path to quality_report.json. Defaults to config value.
    master_df : pd.DataFrame, optional
        The master_metadata DataFrame for resolution consistency analysis.
    class_gini : float, optional
        Gini coefficient from ClassAnalyzer (used in quality score).
    """

    def __init__(
        self,
        quality_report_path: Optional[Path] = None,
        master_df: Optional[pd.DataFrame] = None,
        class_gini: float = 0.0,
    ) -> None:
        self._report_path = Path(quality_report_path) if quality_report_path else QUALITY_REPORT_JSON
        self._master_df = master_df
        self._class_gini = class_gini
        self._report: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}

    def load(self) -> "QualityAnalyzer":
        """Load quality_report.json if it exists."""
        if self._report_path.exists():
            with open(self._report_path, "r", encoding="utf-8") as fh:
                self._report = json.load(fh)
            logger.info("Loaded quality_report.json: %d images", self._report.get("total_images", 0))
        else:
            logger.warning(
                "quality_report.json not found at %s — quality analysis will be limited.",
                self._report_path,
            )
        return self

    def analyze(self) -> Dict[str, Any]:
        """
        Compute quality statistics and Dataset Quality Score.

        Returns
        -------
        dict
            Quality statistics and composite quality score.
        """
        if not self._report and self._report_path.exists():
            self.load()

        logger.info("Running QualityAnalyzer ...")

        # --- From quality_report.json ---
        total_images  = self._report.get("total_images", 0)
        total_flagged = self._report.get("total_flagged", 0)
        flag_rate_pct = self._report.get("flag_rate_pct", 0.0)
        flag_counts   = self._report.get("flag_counts", {})
        per_modality  = self._report.get("per_modality", {})

        # --- Flag breakdown ---
        flag_breakdown: Dict[str, Any] = {}
        for flag, count in flag_counts.items():
            flag_breakdown[flag] = {
                "count": count,
                "pct": round(count / total_images * 100, 2) if total_images > 0 else 0.0,
            }

        # --- Per-modality quality ---
        modality_quality: Dict[str, Any] = {}
        for mod, data in per_modality.items():
            m_total   = data.get("total", 0)
            m_flagged = data.get("flagged", 0)
            m_flags   = data.get("flags", {})
            modality_quality[mod] = {
                "total": m_total,
                "flagged": m_flagged,
                "flag_rate_pct": round(m_flagged / m_total * 100, 2) if m_total > 0 else 0.0,
                "flags": m_flags,
            }

        # --- Resolution consistency (from master_df) ---
        resolution_score_pct, resolution_detail = self._resolution_consistency()

        # --- Duplicate detection via md5 (from master_df) ---
        duplicate_info = self._detect_duplicates()

        # --- Compute quality scores ---
        scores = self._compute_quality_score(
            flag_rate_pct=flag_rate_pct,
            resolution_pct=resolution_score_pct,
        )

        self._stats = {
            "total_images": total_images,
            "total_flagged": total_flagged,
            "overall_flag_rate_pct": flag_rate_pct,
            "flag_breakdown": flag_breakdown,
            "per_modality_quality": modality_quality,
            "resolution_consistency": resolution_detail,
            "duplicates": duplicate_info,
            "quality_score": scores,
        }

        logger.info(
            "QualityAnalyzer complete: flag_rate=%.1f%% | quality_score=%.1f/100",
            flag_rate_pct,
            scores.get("total_score", 0),
        )
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolution_consistency(self) -> tuple:
        """
        Compute the % of images within ±20% of the median dimension.

        Returns
        -------
        (float, dict)
            Percentage consistent (0-100) and detail dict.
        """
        if self._master_df is None:
            return 100.0, {"note": "master_df not provided"}

        df = self._master_df
        if COL_WIDTH not in df.columns or COL_HEIGHT not in df.columns:
            return 100.0, {"note": "width/height columns missing"}

        valid = df[(df[COL_WIDTH] > 0) & (df[COL_HEIGHT] > 0)]
        if valid.empty:
            return 100.0, {"note": "no valid resolution data"}

        median_w = float(valid[COL_WIDTH].median())
        median_h = float(valid[COL_HEIGHT].median())
        lo_w, hi_w = median_w * 0.8, median_w * 1.2
        lo_h, hi_h = median_h * 0.8, median_h * 1.2

        consistent = valid[
            valid[COL_WIDTH].between(lo_w, hi_w) &
            valid[COL_HEIGHT].between(lo_h, hi_h)
        ]
        pct = round(len(consistent) / len(valid) * 100, 2)

        # Per-modality consistency
        per_mod: Dict[str, float] = {}
        if COL_MODALITY in valid.columns:
            for mod, grp in valid.groupby(COL_MODALITY):
                med_w = float(grp[COL_WIDTH].median())
                med_h = float(grp[COL_HEIGHT].median())
                inside = grp[
                    grp[COL_WIDTH].between(med_w * 0.8, med_w * 1.2) &
                    grp[COL_HEIGHT].between(med_h * 0.8, med_h * 1.2)
                ]
                per_mod[mod] = round(len(inside) / len(grp) * 100, 2) if len(grp) > 0 else 100.0

        return pct, {
            "median_width": round(median_w, 1),
            "median_height": round(median_h, 1),
            "consistent_pct": pct,
            "per_modality_pct": per_mod,
        }

    def _detect_duplicates(self) -> Dict[str, Any]:
        """Detect exact duplicates via MD5 hash if available in master_df."""
        if self._master_df is None or "md5_hash" not in self._master_df.columns:
            return {"note": "md5_hash column not available"}

        hashes = self._master_df["md5_hash"].dropna()
        duplicates = hashes[hashes.duplicated(keep=False)]
        return {
            "total_with_hash": int(len(hashes)),
            "duplicate_count": int(len(duplicates)),
            "unique_duplicated_hashes": int(duplicates.nunique()),
            "duplicate_pct": round(len(duplicates) / len(hashes) * 100, 2) if len(hashes) > 0 else 0.0,
        }

    def _compute_quality_score(
        self,
        flag_rate_pct: float,
        resolution_pct: float,
    ) -> Dict[str, Any]:
        """
        Compute the composite Dataset Quality Score.

        1. Class Balance Score  (0–25): 25 × (1 − gini)
        2. Image Quality Score  (0–25): 25 × (1 − flag_rate / 100)
        3. Corruption Score     (0–25): fixed 25 (no corruption in preprocessed)
        4. Resolution Score     (0–25): 25 × (resolution_pct / 100)
        """
        # 1. Class balance
        class_balance_score = round(WEIGHT_CLASS_BALANCE * (1.0 - self._class_gini), 2)

        # 2. Image quality (inverse of flag rate)
        image_quality_score = round(
            WEIGHT_IMAGE_QUALITY * (1.0 - flag_rate_pct / 100.0), 2
        )

        # 3. Corruption: assume processed images have 0% corruption
        corruption_score = float(WEIGHT_CORRUPTION)

        # 4. Resolution consistency
        resolution_score = round(WEIGHT_RESOLUTION * (resolution_pct / 100.0), 2)

        total = round(
            class_balance_score + image_quality_score + corruption_score + resolution_score, 2
        )

        # Grade
        if total >= 85:   grade = "A (Excellent)"
        elif total >= 70: grade = "B (Good)"
        elif total >= 55: grade = "C (Fair)"
        elif total >= 40: grade = "D (Poor)"
        else:             grade = "F (Very Poor)"

        return {
            "class_balance_score": class_balance_score,
            "image_quality_score": image_quality_score,
            "corruption_score": corruption_score,
            "resolution_consistency_score": resolution_score,
            "total_score": total,
            "grade": grade,
            "max_score": 100,
            "weights": {
                "class_balance": WEIGHT_CLASS_BALANCE,
                "image_quality": WEIGHT_IMAGE_QUALITY,
                "corruption": WEIGHT_CORRUPTION,
                "resolution_consistency": WEIGHT_RESOLUTION,
            },
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    @property
    def quality_score(self) -> float:
        """Return the total quality score (0–100)."""
        return self._stats.get("quality_score", {}).get("total_score", 0.0)
