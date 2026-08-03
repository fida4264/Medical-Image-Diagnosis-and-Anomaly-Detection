"""
statistics.py
=============
Computes and serialises comprehensive dataset statistics.

Produces dataset_statistics.json, a machine-readable summary that gives a
complete quantitative picture of the dataset before any model training begins.

The JSON structure is hierarchical:
    {
        "generated_at": "ISO-8601 timestamp",
        "overall": {
            "total_images": int,
            "total_valid_images": int,
            "total_invalid_images": int,
            "total_modalities": int,
            "total_classes": int,
            "total_splits": list[str],
            "total_file_size_bytes": int,
            "total_file_size_mb": float
        },
        "per_modality": {
            "<modality_name>": {
                "total_images": int,
                "valid_images": int,
                "classes": list[str],
                "num_classes": int,
                "splits": list[str],
                "avg_width_px": float,
                "avg_height_px": float,
                "min_width_px": int,
                "max_width_px": int,
                "min_height_px": int,
                "max_height_px": int,
                "avg_file_size_bytes": float,
                "min_file_size_bytes": int,
                "max_file_size_bytes": int,
                "per_class": { "<class>": int, … },
                "per_split": { "<split>": int, … }
            },
            …
        }
    }

Classes
-------
StatisticsGenerator   Computes statistics and writes dataset_statistics.json.

Author: Medical AI System — Module 1
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from dataset_manager.config import (
    COL_DISEASE_CLASS,
    COL_FILE_SIZE_BYTES,
    COL_HEIGHT,
    COL_IS_VALID,
    COL_MODALITY,
    COL_SPLIT,
    COL_WIDTH,
    OUTPUTS_DIR,
    STATISTICS_JSON,
)

logger = logging.getLogger(__name__)


class StatisticsGenerator:
    """
    Derives comprehensive dataset statistics from the metadata DataFrame
    and writes them to dataset_statistics.json.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        The DataFrame produced by MetadataGenerator.generate().
    outputs_dir : Path, optional
        Directory where dataset_statistics.json will be written.

    Examples
    --------
    >>> gen = StatisticsGenerator(metadata_df)
    >>> stats = gen.compute()
    >>> gen.save()
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._meta = metadata_df.copy()
        self._outputs_dir = outputs_dir or OUTPUTS_DIR
        self._stats: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self) -> Dict[str, Any]:
        """
        Compute the full statistics dictionary.

        Returns
        -------
        dict
            Nested statistics dictionary (JSON-serialisable).
        """
        logger.info("Computing dataset statistics …")
        df = self._meta

        overall = self._compute_overall(df)
        per_modality = {}

        for modality in sorted(df[COL_MODALITY].unique()):
            mdf = df[df[COL_MODALITY] == modality]
            per_modality[modality] = self._compute_per_modality(mdf)

        self._stats = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "overall": overall,
            "per_modality": per_modality,
        }
        logger.info("Statistics computation complete.")
        return self._stats

    def save(self, output_path: Optional[Path] = None) -> Path:
        """
        Write the statistics dictionary to JSON.

        Parameters
        ----------
        output_path : Path, optional
            Custom destination; defaults to OUTPUTS_DIR/dataset_statistics.json.

        Returns
        -------
        Path
            The path where the JSON file was written.

        Raises
        ------
        RuntimeError
            When called before compute().
        """
        if self._stats is None:
            raise RuntimeError("Call compute() before save().")

        dest = output_path or (self._outputs_dir / STATISTICS_JSON)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(self._stats, fh, indent=2, default=self._json_default)
        logger.info("dataset_statistics.json saved → %s", dest)
        return dest

    @property
    def statistics(self) -> Optional[Dict[str, Any]]:
        """The computed statistics dictionary, or None if compute() not called."""
        return self._stats

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _compute_overall(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute project-wide aggregate statistics."""
        total = len(df)
        valid = int(df[COL_IS_VALID].sum()) if COL_IS_VALID in df.columns else total
        total_bytes = int(df[COL_FILE_SIZE_BYTES].sum())

        return {
            "total_images":           total,
            "total_valid_images":     valid,
            "total_invalid_images":   total - valid,
            "total_modalities":       int(df[COL_MODALITY].nunique()),
            "total_classes":          int(df[COL_DISEASE_CLASS].nunique()),
            "total_splits":           sorted(df[COL_SPLIT].unique().tolist()),
            "total_file_size_bytes":  total_bytes,
            "total_file_size_mb":     round(total_bytes / 1_000_000, 2),
        }

    def _compute_per_modality(self, mdf: pd.DataFrame) -> Dict[str, Any]:
        """Compute statistics for one modality sub-DataFrame."""
        valid_mdf = (
            mdf[mdf[COL_IS_VALID].astype(bool)]
            if COL_IS_VALID in mdf.columns
            else mdf
        )

        # Dimension stats (exclude zero values from images that could not be read)
        widths = mdf[mdf[COL_WIDTH] > 0][COL_WIDTH]
        heights = mdf[mdf[COL_HEIGHT] > 0][COL_HEIGHT]

        def safe_int(series: pd.Series, func: str) -> int:
            if series.empty:
                return 0
            return int(getattr(series, func)())

        def safe_float(series: pd.Series, func: str) -> float:
            if series.empty:
                return 0.0
            return round(float(getattr(series, func)()), 2)

        per_class = (
            mdf.groupby(COL_DISEASE_CLASS)
            .size()
            .sort_index()
            .to_dict()
        )
        per_split = (
            mdf.groupby(COL_SPLIT)
            .size()
            .sort_index()
            .to_dict()
        )

        return {
            "total_images":           len(mdf),
            "valid_images":           len(valid_mdf),
            "classes":                sorted(mdf[COL_DISEASE_CLASS].unique().tolist()),
            "num_classes":            int(mdf[COL_DISEASE_CLASS].nunique()),
            "splits":                 sorted(mdf[COL_SPLIT].unique().tolist()),
            "avg_width_px":           safe_float(widths, "mean"),
            "avg_height_px":          safe_float(heights, "mean"),
            "min_width_px":           safe_int(widths, "min"),
            "max_width_px":           safe_int(widths, "max"),
            "min_height_px":          safe_int(heights, "min"),
            "max_height_px":          safe_int(heights, "max"),
            "avg_file_size_bytes":    safe_float(mdf[COL_FILE_SIZE_BYTES], "mean"),
            "min_file_size_bytes":    safe_int(mdf[COL_FILE_SIZE_BYTES], "min"),
            "max_file_size_bytes":    safe_int(mdf[COL_FILE_SIZE_BYTES], "max"),
            "per_class":              {str(k): int(v) for k, v in per_class.items()},
            "per_split":              {str(k): int(v) for k, v in per_split.items()},
        }

    @staticmethod
    def _json_default(obj: Any) -> Any:
        """
        JSON serialisation fallback for types not natively supported by json.dump.
        Handles numpy integers, numpy floats, and Path objects.
        """
        try:
            import numpy as np
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        if isinstance(obj, Path):
            return str(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serialisable.")
