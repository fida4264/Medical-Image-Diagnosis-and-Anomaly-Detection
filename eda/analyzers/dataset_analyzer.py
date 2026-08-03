"""
eda/analyzers/dataset_analyzer.py
====================================
Computes overall dataset composition statistics from master_metadata.csv
and dataset_registry.json (Module 2 outputs).

Statistics Generated
---------------------
- Total images, modalities, disease classes
- Images per modality
- Images per normalized disease class
- Train / validation / test split distribution
- Images per dataset source
- Image format (extension) distribution
- Missing data summary

Classes
-------
DatasetAnalyzer
    Loads metadata and computes dataset composition statistics.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from eda.config import (
    MASTER_METADATA_CSV,
    DATASET_REGISTRY_JSON,
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_DATASET_NAME,
    COL_SPLIT,
    COL_EXTENSION,
    COL_IMAGE_PATH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_FILE_SIZE,
    COL_CHANNELS,
    MODALITY_DISPLAY_NAMES,
)

logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    """
    Computes overall dataset composition statistics.

    Parameters
    ----------
    master_csv : Path, optional
        Path to master_metadata.csv.  Defaults to config value.
    registry_json : Path, optional
        Path to dataset_registry.json.  Defaults to config value.
    """

    def __init__(
        self,
        master_csv: Optional[Path] = None,
        registry_json: Optional[Path] = None,
    ) -> None:
        self._master_csv = Path(master_csv) if master_csv else MASTER_METADATA_CSV
        self._registry_json = Path(registry_json) if registry_json else DATASET_REGISTRY_JSON
        self._df: Optional[pd.DataFrame] = None
        self._registry: Dict[str, Any] = {}
        self._stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "DatasetAnalyzer":
        """Load master_metadata.csv and dataset_registry.json."""
        if not self._master_csv.exists():
            raise FileNotFoundError(
                f"master_metadata.csv not found: {self._master_csv}\n"
                "Run Module 2 (run_metadata_manager.py) first."
            )
        self._df = pd.read_csv(self._master_csv, encoding="utf-8", low_memory=False)
        logger.info("Loaded master_metadata.csv: %d rows", len(self._df))

        if self._registry_json.exists():
            with open(self._registry_json, "r", encoding="utf-8") as fh:
                self._registry = json.load(fh)
            logger.info("Loaded dataset_registry.json")
        else:
            logger.warning("dataset_registry.json not found — registry stats will be empty.")
        return self

    def analyze(self) -> Dict[str, Any]:
        """
        Compute all dataset composition statistics.

        Returns
        -------
        dict
            Comprehensive statistics dictionary.
        """
        if self._df is None:
            self.load()

        df = self._df
        logger.info("Running DatasetAnalyzer ...")

        # --- Basic counts ---
        total_images = len(df)
        total_modalities = df[COL_MODALITY].nunique() if COL_MODALITY in df.columns else 0
        total_classes = (
            df[COL_NORMALIZED_CLASS].nunique()
            if COL_NORMALIZED_CLASS in df.columns else 0
        )

        # --- Per-modality ---
        modality_counts: Dict[str, int] = {}
        if COL_MODALITY in df.columns:
            modality_counts = df[COL_MODALITY].value_counts().to_dict()

        # --- Per disease class ---
        class_counts: Dict[str, int] = {}
        if COL_NORMALIZED_CLASS in df.columns:
            class_counts = df[COL_NORMALIZED_CLASS].value_counts().to_dict()

        # --- Split distribution ---
        split_counts: Dict[str, int] = {}
        if COL_SPLIT in df.columns:
            split_counts = df[COL_SPLIT].fillna("unknown").value_counts().to_dict()

        # --- Dataset source ---
        source_counts: Dict[str, int] = {}
        if COL_DATASET_NAME in df.columns:
            source_counts = df[COL_DATASET_NAME].value_counts().to_dict()

        # --- Format distribution ---
        format_counts: Dict[str, int] = {}
        if COL_EXTENSION in df.columns:
            format_counts = (
                df[COL_EXTENSION].str.lower().value_counts().to_dict()
            )

        # --- Missing data ---
        missing: Dict[str, int] = {
            col: int(df[col].isna().sum())
            for col in df.columns
            if df[col].isna().any()
        }

        # --- Per-modality class diversity ---
        modality_class_diversity: Dict[str, int] = {}
        if COL_MODALITY in df.columns and COL_NORMALIZED_CLASS in df.columns:
            modality_class_diversity = (
                df.groupby(COL_MODALITY)[COL_NORMALIZED_CLASS]
                .nunique()
                .to_dict()
            )

        # --- Display names ---
        modality_display = {
            mod: MODALITY_DISPLAY_NAMES.get(mod, mod.replace("_", " ").title())
            for mod in modality_counts
        }

        # --- Resolution summary ---
        res_summary: Dict[str, Any] = {}
        if COL_WIDTH in df.columns and COL_HEIGHT in df.columns:
            valid = df[(df[COL_WIDTH] > 0) & (df[COL_HEIGHT] > 0)]
            if not valid.empty:
                res_summary = {
                    "width_min": int(valid[COL_WIDTH].min()),
                    "width_max": int(valid[COL_WIDTH].max()),
                    "width_mean": round(float(valid[COL_WIDTH].mean()), 1),
                    "width_median": round(float(valid[COL_WIDTH].median()), 1),
                    "height_min": int(valid[COL_HEIGHT].min()),
                    "height_max": int(valid[COL_HEIGHT].max()),
                    "height_mean": round(float(valid[COL_HEIGHT].mean()), 1),
                    "height_median": round(float(valid[COL_HEIGHT].median()), 1),
                }

        # --- File size summary ---
        size_summary: Dict[str, Any] = {}
        if COL_FILE_SIZE in df.columns:
            fs = df[COL_FILE_SIZE].dropna()
            if not fs.empty:
                size_summary = {
                    "min_bytes": int(fs.min()),
                    "max_bytes": int(fs.max()),
                    "mean_bytes": round(float(fs.mean()), 1),
                    "median_bytes": round(float(fs.median()), 1),
                    "total_gb": round(float(fs.sum()) / (1024 ** 3), 3),
                }

        # --- Registry info ---
        registry_datasets = len(self._registry.get("datasets", {})) if self._registry else 0

        self._stats = {
            "total_images": total_images,
            "total_modalities": total_modalities,
            "total_disease_classes": total_classes,
            "total_dataset_sources": len(source_counts),
            "images_per_modality": modality_counts,
            "modality_display_names": modality_display,
            "images_per_class": class_counts,
            "split_distribution": split_counts,
            "images_per_source": source_counts,
            "format_distribution": format_counts,
            "modality_class_diversity": modality_class_diversity,
            "resolution_summary": res_summary,
            "file_size_summary": size_summary,
            "missing_data": missing,
            "registry_dataset_count": registry_datasets,
        }

        logger.info(
            "DatasetAnalyzer complete: %d images | %d modalities | %d classes",
            total_images, total_modalities, total_classes,
        )
        return self._stats

    @property
    def stats(self) -> Dict[str, Any]:
        """Return the computed statistics dict (empty if analyze() not called)."""
        return self._stats

    @property
    def df(self) -> Optional[pd.DataFrame]:
        """The loaded master_metadata DataFrame."""
        return self._df

    @property
    def registry(self) -> Dict[str, Any]:
        """The loaded dataset_registry.json content."""
        return self._registry

    def summary_lines(self) -> List[str]:
        """Return a list of human-readable summary lines."""
        s = self._stats
        lines = [
            f"Total Images         : {s.get('total_images', 0):,}",
            f"Total Modalities     : {s.get('total_modalities', 0)}",
            f"Total Disease Classes: {s.get('total_disease_classes', 0)}",
            f"Dataset Sources      : {s.get('total_dataset_sources', 0)}",
        ]
        for mod, cnt in s.get("images_per_modality", {}).items():
            display = MODALITY_DISPLAY_NAMES.get(mod, mod)
            lines.append(f"  {display:<22}: {cnt:,}")
        return lines
