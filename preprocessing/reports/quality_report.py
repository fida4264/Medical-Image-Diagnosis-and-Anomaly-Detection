"""
preprocessing/reports/quality_report.py
==========================================
Accumulates per-image quality flags and writes quality_report.json.

Structure of quality_report.json
---------------------------------
{
  "generated_at": "...",
  "total_images": 118715,
  "total_flagged": 312,
  "flag_counts": {
    "blurry": 102,
    "too_dark": 45,
    ...
  },
  "per_modality": {
    "brain_mri": { "total": 7200, "flagged": 12, "flags": {...} },
    ...
  },
  "flagged_images": [
    { "internal_id": "MRI_000042", "flags": ["blurry"] },
    ...
  ]
}

Classes
-------
QualityReportWriter
    Accumulates quality records and saves the JSON report.

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

logger = logging.getLogger(__name__)


class QualityReportWriter:
    """
    Thread-safe accumulator for image quality flags.

    Usage
    -----
        writer = QualityReportWriter()
        writer.record("MRI_000001", "brain_mri", ["blurry"])
        writer.record("MRI_000002", "brain_mri", [])
        writer.save(Path("processed_images/quality_report.json"))
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        self._total = 0
        self._flagged = 0
        self._flag_counts: Dict[str, int] = defaultdict(int)
        self._per_modality: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "flagged": 0, "flags": defaultdict(int)}
        )

    def record(
        self,
        internal_id: str,
        modality: str,
        flags: List[str],
    ) -> None:
        """
        Record the quality flags for one image.

        Parameters
        ----------
        internal_id : str
            Unique image ID (e.g., 'MRI_000042').
        modality : str
            Modality string (e.g., 'brain_mri').
        flags : list of str
            Quality flags detected.  Empty = image passed all checks.
        """
        with self._lock:
            self._total += 1
            mod = self._per_modality[modality]
            mod["total"] += 1

            if flags:
                self._flagged += 1
                mod["flagged"] += 1
                self._records.append({
                    "internal_id": internal_id,
                    "modality": modality,
                    "flags": flags,
                })
                for f in flags:
                    self._flag_counts[f] += 1
                    mod["flags"][f] += 1

    def save(self, output_path: Path) -> Path:
        """
        Write quality_report.json to disk.

        Parameters
        ----------
        output_path : Path
            Destination file path.

        Returns
        -------
        Path
            The saved file path.
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdicts for JSON serialisation
        per_modality_clean = {}
        for mod, data in self._per_modality.items():
            per_modality_clean[mod] = {
                "total": data["total"],
                "flagged": data["flagged"],
                "flag_rate_pct": (
                    round(data["flagged"] / data["total"] * 100, 2)
                    if data["total"] > 0
                    else 0.0
                ),
                "flags": dict(data["flags"]),
            }

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_images": self._total,
            "total_flagged": self._flagged,
            "flag_rate_pct": (
                round(self._flagged / self._total * 100, 2) if self._total > 0 else 0.0
            ),
            "flag_counts": dict(self._flag_counts),
            "per_modality": per_modality_clean,
            "flagged_images": self._records,
        }

        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, default=str)

        logger.info(
            "quality_report.json saved -> %s  (%d/%d flagged, %.1f%%)",
            output_path,
            self._flagged,
            self._total,
            report["flag_rate_pct"],
        )
        return output_path

    @property
    def total(self) -> int:
        return self._total

    @property
    def flagged(self) -> int:
        return self._flagged
