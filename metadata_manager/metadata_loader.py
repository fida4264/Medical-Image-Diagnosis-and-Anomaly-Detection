"""
metadata_manager/metadata_loader.py
=====================================
Loads and validates the five output files produced by Module 1.

This module is the single entry point for all Module 1 data. Every other
Module 2 component receives its raw data from MetadataLoader — never by
reading files directly.

Classes
-------
MetadataLoader
    Reads metadata.csv, labels.csv, class_distribution.csv,
    dataset_statistics.json, and validation_report.json.
    Validates that all required columns are present.
    Exposes clean, typed DataFrames and dicts as properties.

Usage
-----
    loader = MetadataLoader()
    loader.load()
    meta_df   = loader.metadata_df
    labels_df = loader.labels_df
    stats     = loader.statistics

Author: Medical AI System — Module 2
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from metadata_manager.config import (
    METADATA_CSV,
    LABELS_CSV,
    STATISTICS_JSON,
    VALIDATION_REPORT_JSON,
    CLASS_DISTRIBUTION_CSV,
    MODULE1_OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Required columns per file (schema contract with Module 1)
# ---------------------------------------------------------------------------

REQUIRED_METADATA_COLS = {
    "image_id", "modality", "disease_class", "split",
    "image_path", "filename", "extension", "width",
    "height", "file_size_bytes", "is_valid", "md5_hash",
}

REQUIRED_LABELS_COLS = {
    "image_id", "modality", "disease_class", "split", "label", "class_name",
}

REQUIRED_DISTRIBUTION_COLS = {"modality", "disease_class", "split", "count"}


class MetadataLoader:
    """
    Loads and validates all Module 1 output files.

    Parameters
    ----------
    outputs_dir : Path, optional
        Directory containing Module 1 outputs.
        Defaults to the value from config.py.

    Raises
    ------
    FileNotFoundError
        If metadata.csv or labels.csv are missing.
    ValueError
        If any required column is absent from a loaded file.
    """

    def __init__(self, outputs_dir: Optional[Path] = None) -> None:
        self._dir: Path = Path(outputs_dir) if outputs_dir else MODULE1_OUTPUTS_DIR
        self._metadata_df: Optional[pd.DataFrame] = None
        self._labels_df: Optional[pd.DataFrame] = None
        self._distribution_df: Optional[pd.DataFrame] = None
        self._statistics: Optional[Dict[str, Any]] = None
        self._validation_report: Optional[Dict[str, Any]] = None
        self._loaded: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> "MetadataLoader":
        """
        Load all Module 1 output files from disk.

        Returns
        -------
        MetadataLoader
            Self, for method chaining.
        """
        logger.info("Loading Module 1 outputs from: %s", self._dir)
        self._metadata_df = self._load_metadata_csv()
        self._labels_df = self._load_labels_csv()
        self._distribution_df = self._load_distribution_csv()
        self._statistics = self._load_json(self._dir / "dataset_statistics.json")
        self._validation_report = self._load_json(self._dir / "validation_report.json")
        self._loaded = True
        logger.info(
            "Load complete. metadata=%d rows | labels=%d rows",
            len(self._metadata_df),
            len(self._labels_df),
        )
        return self

    @property
    def metadata_df(self) -> pd.DataFrame:
        """Full metadata DataFrame from Module 1 (118 715 rows × 12 cols)."""
        self._assert_loaded()
        return self._metadata_df  # type: ignore[return-value]

    @property
    def labels_df(self) -> pd.DataFrame:
        """Labels DataFrame from Module 1."""
        self._assert_loaded()
        return self._labels_df  # type: ignore[return-value]

    @property
    def distribution_df(self) -> pd.DataFrame:
        """Class distribution DataFrame from Module 1."""
        self._assert_loaded()
        return self._distribution_df  # type: ignore[return-value]

    @property
    def statistics(self) -> Dict[str, Any]:
        """Dataset statistics dict from Module 1."""
        self._assert_loaded()
        return self._statistics  # type: ignore[return-value]

    @property
    def validation_report(self) -> Dict[str, Any]:
        """Validation report dict from Module 1."""
        self._assert_loaded()
        return self._validation_report  # type: ignore[return-value]

    @property
    def is_loaded(self) -> bool:
        """True if load() has been called successfully."""
        return self._loaded

    def summary(self) -> str:
        """Return a one-line human-readable summary of loaded data."""
        self._assert_loaded()
        mods = self._metadata_df["modality"].nunique()  # type: ignore[union-attr]
        valid = int(self._metadata_df["is_valid"].sum())  # type: ignore[union-attr]
        total = len(self._metadata_df)  # type: ignore[union-attr]
        return (
            f"MetadataLoader: {total:,} images | {mods} modalities | "
            f"{valid:,} valid ({valid / total:.1%})"
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_metadata_csv(self) -> pd.DataFrame:
        path = self._dir / "metadata.csv"
        self._assert_file_exists(path)
        df = pd.read_csv(path, low_memory=False)
        self._validate_columns(df, REQUIRED_METADATA_COLS, "metadata.csv")
        df["is_valid"] = df["is_valid"].astype(bool)
        df["width"] = pd.to_numeric(df["width"], errors="coerce").fillna(0).astype(int)
        df["height"] = pd.to_numeric(df["height"], errors="coerce").fillna(0).astype(int)
        df["file_size_bytes"] = pd.to_numeric(
            df["file_size_bytes"], errors="coerce"
        ).fillna(0).astype(int)
        logger.info("  metadata.csv loaded: %d rows", len(df))
        return df

    def _load_labels_csv(self) -> pd.DataFrame:
        path = self._dir / "labels.csv"
        self._assert_file_exists(path)
        df = pd.read_csv(path, low_memory=False)
        self._validate_columns(df, REQUIRED_LABELS_COLS, "labels.csv")
        df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)
        logger.info("  labels.csv loaded: %d rows", len(df))
        return df

    def _load_distribution_csv(self) -> pd.DataFrame:
        path = self._dir / "class_distribution.csv"
        if not path.exists():
            logger.warning("class_distribution.csv not found — skipping.")
            return pd.DataFrame(columns=list(REQUIRED_DISTRIBUTION_COLS))
        df = pd.read_csv(path)
        self._validate_columns(df, REQUIRED_DISTRIBUTION_COLS, "class_distribution.csv")
        logger.info("  class_distribution.csv loaded: %d rows", len(df))
        return df

    def _load_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            logger.warning("JSON file not found: %s — returning empty dict.", path.name)
            return {}
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("  %s loaded.", path.name)
        return data

    @staticmethod
    def _assert_file_exists(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"Required Module 1 output not found: {path}\n"
                f"Run 'python run_dataset_manager.py' first."
            )

    @staticmethod
    def _validate_columns(
        df: pd.DataFrame, required: set, filename: str
    ) -> None:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Schema error in {filename}: missing columns {sorted(missing)}"
            )

    def _assert_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError(
                "MetadataLoader.load() must be called before accessing data."
            )
