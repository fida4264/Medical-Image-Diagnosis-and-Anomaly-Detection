"""
preprocessing/reports/metadata_writer.py
==========================================
Accumulates per-image preprocessing records and writes:
  - preprocessing_metadata.csv  (full per-image history)
  - preprocessing_report.json   (summary statistics)

CSV Schema (preprocessing_metadata.csv)
-----------------------------------------
image_id, internal_id, original_path, processed_path,
modality, normalized_class, operations_applied,
original_width, original_height, processed_width, processed_height,
processing_time_s, status, quality_flags

Classes
-------
MetadataWriter
    Thread-safe accumulator + CSV/JSON writer.

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

METADATA_COLUMNS = [
    "image_id",
    "internal_id",
    "original_path",
    "processed_path",
    "modality",
    "normalized_class",
    "operations_applied",
    "original_width",
    "original_height",
    "processed_width",
    "processed_height",
    "processing_time_s",
    "status",
    "quality_flags",
]


class MetadataWriter:
    """
    Thread-safe writer for preprocessing history.

    Usage
    -----
        writer = MetadataWriter()
        writer.record(**row_dict)
        writer.save_csv(Path("processed_images/preprocessing_metadata.csv"))
        writer.save_report(Path("processed_images/preprocessing_report.json"))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._rows: List[Dict[str, Any]] = []
        self._counts: Dict[str, int] = defaultdict(int)   # status counts
        self._mod_counts: Dict[str, int] = defaultdict(int)
        self._total_time: float = 0.0

    def record(
        self,
        image_id: str,
        internal_id: str,
        original_path: str,
        processed_path: str,
        modality: str,
        normalized_class: str,
        operations_applied: List[str],
        original_shape: tuple,
        processed_shape: tuple,
        processing_time_s: float,
        status: str,
        quality_flags: List[str],
    ) -> None:
        """
        Record the outcome of preprocessing one image.

        Parameters
        ----------
        image_id : str
            Original Module 2 image_id.
        internal_id : str
            Unique internal ID (e.g., 'MRI_000042').
        original_path : str
            Path to the source image.
        processed_path : str
            Path where the processed image was saved.
        modality : str
            Modality string.
        normalized_class : str
            Normalized disease class.
        operations_applied : list of str
            Ordered list of operations performed.
        original_shape : tuple
            (height, width[, channels]) of the original image.
        processed_shape : tuple
            (height, width, channels) of the processed image.
        processing_time_s : float
            Wall-clock time taken to process this image.
        status : str
            'success' or error description.
        quality_flags : list of str
            Quality flags detected on the original image.
        """
        orig_h, orig_w = original_shape[:2]
        proc_h, proc_w = processed_shape[:2]

        row = {
            "image_id": image_id,
            "internal_id": internal_id,
            "original_path": original_path,
            "processed_path": processed_path,
            "modality": modality,
            "normalized_class": normalized_class,
            "operations_applied": "|".join(operations_applied),
            "original_width": orig_w,
            "original_height": orig_h,
            "processed_width": proc_w,
            "processed_height": proc_h,
            "processing_time_s": round(processing_time_s, 5),
            "status": status,
            "quality_flags": "|".join(quality_flags) if quality_flags else "",
        }

        with self._lock:
            self._rows.append(row)
            self._counts[status] += 1
            self._mod_counts[modality] += 1
            self._total_time += processing_time_s

    def save_csv(self, output_path: Path) -> Path:
        """
        Write preprocessing_metadata.csv.

        Parameters
        ----------
        output_path : Path
            Destination CSV file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self._rows, columns=METADATA_COLUMNS)
        df.to_csv(output_path, index=False, encoding="utf-8")
        logger.info(
            "preprocessing_metadata.csv saved -> %s  (%d rows)",
            output_path, len(df),
        )
        return output_path

    def save_report(self, output_path: Path) -> Path:
        """
        Write preprocessing_report.json with summary statistics.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total = len(self._rows)
        success = self._counts.get("success", 0)
        errors = total - success

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_processed": total,
            "successful": success,
            "errors": errors,
            "success_rate_pct": round(success / total * 100, 2) if total > 0 else 0.0,
            "total_processing_time_s": round(self._total_time, 2),
            "avg_processing_time_ms": (
                round(self._total_time / total * 1000, 2) if total > 0 else 0.0
            ),
            "by_modality": dict(self._mod_counts),
            "by_status": dict(self._counts),
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)
        logger.info("preprocessing_report.json saved -> %s", output_path)
        return output_path

    @property
    def total_processed(self) -> int:
        return len(self._rows)

    @property
    def total_errors(self) -> int:
        return self._counts.get("error", 0)
