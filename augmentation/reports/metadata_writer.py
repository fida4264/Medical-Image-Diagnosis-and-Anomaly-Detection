"""
augmentation/reports/metadata_writer.py
=========================================
Thread-safe accumulator for augmentation history.

Writes:
  - augmentation_metadata.csv   (full per-image augmentation record)
  - augmentation_report.json    (summary statistics)

CSV Schema
----------
original_image_id, original_path, augmented_path, modality,
normalized_class, augmentation_index, operations_applied,
seed, timestamp, status

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from augmentation.config import AUGMENTATION_METADATA_COLUMNS

logger = logging.getLogger(__name__)


class AugmentationMetadataWriter:
    """
    Thread-safe writer for augmentation history.

    Usage
    -----
        writer = AugmentationMetadataWriter()
        writer.record(
            original_image_id="MRI_000001",
            original_path="/path/to/original.png",
            augmented_path="/path/to/aug_0.png",
            modality="brain_mri",
            normalized_class="glioma",
            augmentation_index=0,
            operations_applied=["Rotate", "GaussNoise"],
            seed=42,
            status="success",
        )
        writer.save_csv(Path("augmentation_metadata.csv"))
        writer.save_report(Path("augmentation_report.json"))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        self._counts: Dict[str, int] = defaultdict(int)
        self._mod_counts: Dict[str, int] = defaultdict(int)
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._total_originals: int = 0
        self._total_augmented: int = 0

    def record(
        self,
        original_image_id: str,
        original_path: str,
        augmented_path: str,
        modality: str,
        normalized_class: str,
        augmentation_index: int,
        operations_applied: List[str],
        seed: int,
        status: str,
    ) -> None:
        """
        Record one augmented image.

        Parameters
        ----------
        original_image_id : str
            ID of the source image (from preprocessing_metadata.csv).
        original_path : str
            Path to the processed (source) image.
        augmented_path : str
            Path where the augmented image was saved.
        modality : str
            Modality string.
        normalized_class : str
            Normalized disease class.
        augmentation_index : int
            Which augmented copy this is (0, 1, 2, ...).
        operations_applied : list of str
            Transform names applied.
        seed : int
            Random seed used.
        status : str
            'success' or error description.
        """
        row = {
            "original_image_id": original_image_id,
            "original_path": original_path,
            "augmented_path": augmented_path,
            "modality": modality,
            "normalized_class": normalized_class,
            "augmentation_index": augmentation_index,
            "operations_applied": "|".join(operations_applied),
            "seed": seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": status,
        }
        with self._lock:
            self._rows.append(row)
            self._counts[status] += 1
            self._mod_counts[modality] += 1
            key = f"{modality}/{normalized_class}"
            self._class_counts[key] += 1
            if status == "success":
                self._total_augmented += 1

    def set_total_originals(self, count: int) -> None:
        """Set the count of original images processed."""
        with self._lock:
            self._total_originals = count

    def save_csv(self, output_path: Path) -> Path:
        """Write augmentation_metadata.csv."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows, columns=AUGMENTATION_METADATA_COLUMNS)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(
            "augmentation_metadata.csv saved → %s  (%d rows)",
            output_path, len(df),
        )
        return output_path

    def save_report(self, output_path: Path) -> Path:
        """Write augmentation_report.json."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(self._rows)
        success = self._counts.get("success", 0)
        errors = total - success

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_original_images": self._total_originals,
            "total_augmented_images": success,
            "total_errors": errors,
            "augmentation_factor": (
                round(success / self._total_originals, 2)
                if self._total_originals > 0 else 0.0
            ),
            "success_rate_pct": (
                round(success / total * 100, 2) if total > 0 else 0.0
            ),
            "by_modality": dict(self._mod_counts),
            "by_status": dict(self._counts),
            "by_modality_class": dict(self._class_counts),
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("augmentation_report.json saved → %s", output_path)
        return output_path

    @property
    def total_augmented(self) -> int:
        """Total successfully augmented images."""
        return self._total_augmented

    @property
    def total_errors(self) -> int:
        """Total records with any error status (status starts with 'error')."""
        return sum(v for k, v in self._counts.items() if k.startswith("error"))

    @property
    def total_records(self) -> int:
        return len(self._rows)
