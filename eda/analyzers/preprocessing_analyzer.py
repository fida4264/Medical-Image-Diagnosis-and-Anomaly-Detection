"""
eda/analyzers/preprocessing_analyzer.py
==========================================
Analyzes preprocessing outcomes from Module 3's preprocessing_metadata.csv.

Statistics Generated
---------------------
- Success / error rates per modality
- Original vs processed resolution comparison
- Processing time statistics
- Operations applied frequency
- Quality flag frequency before preprocessing

Classes
-------
PreprocessingAnalyzer
    Reads preprocessing_metadata.csv and computes preprocessing analytics.

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
    PREPROCESSING_METADATA_CSV,
    PREPROCESSING_REPORT_JSON,
    COL_MODALITY,
    COL_STATUS,
    COL_PROC_TIME,
    COL_QUALITY_FLAGS,
    COL_ORIGINAL_WIDTH,
    COL_ORIGINAL_HEIGHT,
    COL_PROCESSED_WIDTH,
    COL_PROCESSED_HEIGHT,
    COL_OPERATIONS,
    COL_NORMALIZED_CLASS,
)

logger = logging.getLogger(__name__)


class PreprocessingAnalyzer:
    """
    Analyzes Module 3 preprocessing outcomes.

    Parameters
    ----------
    metadata_csv : Path, optional
        Path to preprocessing_metadata.csv.
    report_json : Path, optional
        Path to preprocessing_report.json.
    """

    def __init__(
        self,
        metadata_csv: Optional[Path] = None,
        report_json: Optional[Path] = None,
    ) -> None:
        self._csv = Path(metadata_csv) if metadata_csv else PREPROCESSING_METADATA_CSV
        self._report_json = Path(report_json) if report_json else PREPROCESSING_REPORT_JSON
        self._df: Optional[pd.DataFrame] = None
        self._report: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}

    def load(self) -> "PreprocessingAnalyzer":
        """Load preprocessing_metadata.csv and preprocessing_report.json."""
        if self._csv.exists():
            self._df = pd.read_csv(self._csv, encoding="utf-8", low_memory=False)
            logger.info("Loaded preprocessing_metadata.csv: %d rows", len(self._df))
        else:
            logger.warning("preprocessing_metadata.csv not found at %s", self._csv)

        if self._report_json.exists():
            with open(self._report_json, "r", encoding="utf-8") as fh:
                self._report = json.load(fh)
        return self

    def analyze(self) -> Dict[str, Any]:
        """
        Compute preprocessing statistics.

        Returns
        -------
        dict
            Preprocessing analytics dictionary.
        """
        if self._df is None:
            self.load()

        if self._df is None or self._df.empty:
            logger.warning("No preprocessing data available.")
            return {"available": False}

        logger.info("Running PreprocessingAnalyzer ...")
        df = self._df

        # --- Status counts ---
        status_counts: Dict[str, int] = {}
        if COL_STATUS in df.columns:
            status_counts = df[COL_STATUS].value_counts().to_dict()

        total = len(df)
        success = status_counts.get("success", 0)
        skipped = status_counts.get("skipped", 0)
        errors = total - success - skipped
        success_rate = round(success / total * 100, 2) if total > 0 else 0.0

        # --- Per-modality status ---
        modality_status: Dict[str, Any] = {}
        if COL_MODALITY in df.columns and COL_STATUS in df.columns:
            for mod, grp in df.groupby(COL_MODALITY):
                sc = grp[COL_STATUS].value_counts().to_dict()
                n = len(grp)
                modality_status[mod] = {
                    "total": n,
                    "success": sc.get("success", 0),
                    "skipped": sc.get("skipped", 0),
                    "errors": n - sc.get("success", 0) - sc.get("skipped", 0),
                    "success_rate_pct": round(sc.get("success", 0) / n * 100, 2) if n > 0 else 0.0,
                }

        # --- Processing time ---
        time_stats: Dict[str, Any] = {}
        if COL_PROC_TIME in df.columns:
            t = df[COL_PROC_TIME].dropna()
            if not t.empty:
                time_stats = {
                    "total_s": round(float(t.sum()), 2),
                    "mean_ms": round(float(t.mean()) * 1000, 2),
                    "median_ms": round(float(t.median()) * 1000, 2),
                    "max_s": round(float(t.max()), 4),
                    "min_ms": round(float(t.min()) * 1000, 4),
                }

        # --- Resolution changes ---
        resolution_change: Dict[str, Any] = {}
        orig_cols = [COL_ORIGINAL_WIDTH, COL_ORIGINAL_HEIGHT]
        proc_cols = [COL_PROCESSED_WIDTH, COL_PROCESSED_HEIGHT]
        if all(c in df.columns for c in orig_cols + proc_cols):
            valid = df[
                (df[COL_ORIGINAL_WIDTH] > 0) & (df[COL_PROCESSED_WIDTH] > 0)
            ]
            if not valid.empty:
                orig_mpx = (valid[COL_ORIGINAL_WIDTH] * valid[COL_ORIGINAL_HEIGHT]) / 1e6
                proc_mpx = (valid[COL_PROCESSED_WIDTH] * valid[COL_PROCESSED_HEIGHT]) / 1e6
                resolution_change = {
                    "original_width_mean": round(float(valid[COL_ORIGINAL_WIDTH].mean()), 1),
                    "original_height_mean": round(float(valid[COL_ORIGINAL_HEIGHT].mean()), 1),
                    "processed_width": int(valid[COL_PROCESSED_WIDTH].mode()[0]) if not valid[COL_PROCESSED_WIDTH].empty else 224,
                    "processed_height": int(valid[COL_PROCESSED_HEIGHT].mode()[0]) if not valid[COL_PROCESSED_HEIGHT].empty else 224,
                    "avg_original_megapixels": round(float(orig_mpx.mean()), 3),
                    "processed_megapixels": round(float(proc_mpx.mean()), 3),
                    "avg_scale_factor": round(float((proc_mpx / orig_mpx).median()), 3),
                }

        # --- Operations frequency ---
        ops_freq: Dict[str, int] = {}
        if COL_OPERATIONS in df.columns:
            for ops_str in df[COL_OPERATIONS].dropna():
                for op in str(ops_str).split("|"):
                    op = op.strip()
                    if op:
                        ops_freq[op] = ops_freq.get(op, 0) + 1
            # Sort by frequency
            ops_freq = dict(sorted(ops_freq.items(), key=lambda x: x[1], reverse=True))

        # --- Quality flags frequency ---
        flags_freq: Dict[str, int] = {}
        if COL_QUALITY_FLAGS in df.columns:
            for flags_str in df[COL_QUALITY_FLAGS].dropna():
                for flag in str(flags_str).split("|"):
                    flag = flag.strip()
                    if flag:
                        flags_freq[flag] = flags_freq.get(flag, 0) + 1

        self._stats = {
            "available": True,
            "total_records": total,
            "success": success,
            "skipped": skipped,
            "errors": errors,
            "success_rate_pct": success_rate,
            "status_counts": status_counts,
            "per_modality_status": modality_status,
            "processing_time": time_stats,
            "resolution_change": resolution_change,
            "operations_frequency": ops_freq,
            "quality_flags_frequency": flags_freq,
            "report_summary": self._report,
        }

        logger.info(
            "PreprocessingAnalyzer complete: %d records | %.1f%% success",
            total, success_rate,
        )
        return self._stats

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats

    @property
    def df(self) -> Optional[pd.DataFrame]:
        return self._df
