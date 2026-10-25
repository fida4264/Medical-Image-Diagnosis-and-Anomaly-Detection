"""
metadata_manager/registry_generator.py
=========================================
Builds dataset_registry.json — a machine-readable catalogue of every
dataset in the system.

Each entry in the registry covers one dataset source (e.g., COVID-19
Radiography Database, BUSI, HAM10000) and includes:
  - name, source, URL, license
  - number of images (total and valid)
  - number of disease classes
  - modality
  - last scan timestamp (from dataset_statistics.json)
  - paper citation (if known)

The registry is used by future modules for dataset provenance and reporting.

Classes
-------
RegistryGenerator
    Builds and saves dataset_registry.json.

Author: Medical AI System — Module 2
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from metadata_manager.config import (
    DATASET_REGISTRY_INFO,
    METADATA_OUTPUT_DIR,
    DATASET_REGISTRY_JSON,
    COL_MODALITY,
    COL_DATASET_NAME,
    COL_NORMALIZED_CLASS,
    COL_IMAGE_ID,
)

logger = logging.getLogger(__name__)


class RegistryGenerator:
    """
    Builds the dataset_registry.json from master_metadata.

    Parameters
    ----------
    master_df : pd.DataFrame
        The standardised master_metadata DataFrame.
    statistics : dict
        Dataset statistics dict from Module 1 (dataset_statistics.json).
    outputs_dir : Path, optional
        Directory where dataset_registry.json will be saved.

    Usage
    -----
        gen = RegistryGenerator(master_df, statistics)
        registry = gen.generate()
        gen.save()
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        statistics: Optional[Dict[str, Any]] = None,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._master_df: pd.DataFrame = master_df
        self._statistics: Dict[str, Any] = statistics or {}
        self._outputs_dir: Path = Path(outputs_dir) if outputs_dir else METADATA_OUTPUT_DIR
        self._registry: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Dict[str, Any]:
        """
        Build the complete dataset registry dict.

        Returns
        -------
        dict
            The full registry, keyed by dataset_source/raw folder name.
        """
        logger.info("Generating dataset registry ...")
        df = self._master_df

        # Group by dataset_name to aggregate per-dataset statistics
        # We use 'original_class' col to identify the raw source key
        source_col = "dataset_source"
        if source_col not in df.columns:
            source_col = COL_DATASET_NAME

        datasets: Dict[str, Any] = {}
        scan_time = self._statistics.get("generated_at", datetime.now(timezone.utc).isoformat())

        # Build registry from known dataset info + actual counts from master_df
        all_raw_keys = list(DATASET_REGISTRY_INFO.keys())

        for raw_key, base_info in DATASET_REGISTRY_INFO.items():
            # Filter master_df to rows belonging to this dataset
            mod = base_info.get("modality", "")
            dname = base_info.get("name", "")
            subset = df[(df[COL_DATASET_NAME] == dname) & (df[COL_MODALITY] == mod)]
            if subset.empty:
                subset = df[df[COL_DATASET_NAME] == dname]
            if subset.empty:
                subset = df[df[source_col] == base_info.get("source", "")]

            total_images = len(subset)
            valid_images = int(subset.get("is_valid", pd.Series([True] * len(subset))).sum()) \
                if "is_valid" in subset.columns else total_images

            classes = (
                sorted(subset[COL_NORMALIZED_CLASS].dropna().unique().tolist())
                if COL_NORMALIZED_CLASS in subset.columns and not subset.empty
                else []
            )
            modality = (
                subset[COL_MODALITY].iloc[0]
                if COL_MODALITY in subset.columns and not subset.empty
                else base_info.get("modality", "unknown")
            )

            datasets[raw_key] = {
                "name": base_info.get("name", raw_key),
                "source": base_info.get("source", "Unknown"),
                "url": base_info.get("url", ""),
                "license": base_info.get("license", "Unknown"),
                "modality": modality,
                "paper": base_info.get("paper", ""),
                "total_images": total_images,
                "valid_images": valid_images,
                "num_classes": len(classes),
                "classes": classes,
                "last_scan_time": scan_time,
            }

        self._registry = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_datasets": len(datasets),
            "total_images": len(df),
            "total_modalities": df[COL_MODALITY].nunique() if COL_MODALITY in df.columns else 0,
            "datasets": datasets,
        }

        logger.info(
            "Registry generated: %d dataset entries.", len(datasets)
        )
        return self._registry

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save dataset_registry.json to disk.

        Returns
        -------
        Path
            Path of the saved file.

        Raises
        ------
        RuntimeError
            If generate() has not been called first.
        """
        if self._registry is None:
            raise RuntimeError("generate() must be called before save().")
        dest_dir = Path(output_dir) if output_dir else self._outputs_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / DATASET_REGISTRY_JSON
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(self._registry, fh, indent=2, default=str)
        logger.info("dataset_registry.json saved -> %s", dest)
        return dest

    @property
    def registry(self) -> Optional[Dict[str, Any]]:
        """The registry dict, or None before generate()."""
        return self._registry
