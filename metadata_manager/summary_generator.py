"""
metadata_manager/summary_generator.py
=======================================
Generates dataset_summary.md — a human-readable statistical report
covering every aspect of the unified dataset.

Sections
--------
1. Overview (total images, modalities, classes, splits)
2. Per-modality breakdown (count, classes, avg resolution)
3. Per-class counts across all modalities
4. Dataset sources (images per dataset)
5. Resolution statistics (min/max/avg width & height)
6. Supported extensions
7. Data quality summary (valid vs invalid images)

Classes
-------
SummaryGenerator
    Builds and saves dataset_summary.md.

Author: Medical AI System — Module 2
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from metadata_manager.config import (
    METADATA_OUTPUT_DIR,
    DATASET_SUMMARY_MD,
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_DATASET_NAME,
    COL_SPLIT,
    COL_WIDTH,
    COL_HEIGHT,
    COL_FILE_SIZE,
    COL_EXTENSION,
)

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """
    Generates a human-readable Markdown summary of the standardised dataset.

    Parameters
    ----------
    master_df : pd.DataFrame
        The standardised master_metadata DataFrame.
    statistics : dict, optional
        Dataset statistics from Module 1.
    outputs_dir : Path, optional
        Directory where dataset_summary.md will be saved.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        statistics: Optional[Dict[str, Any]] = None,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._df: pd.DataFrame = master_df
        self._stats: Dict[str, Any] = statistics or {}
        self._outputs_dir: Path = Path(outputs_dir) if outputs_dir else METADATA_OUTPUT_DIR
        self._content: Optional[str] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> str:
        """
        Build the full Markdown summary string.

        Returns
        -------
        str
            Complete Markdown document.
        """
        logger.info("Generating dataset_summary.md ...")
        df = self._df
        lines = []

        # Header
        lines += [
            "# Dataset Summary",
            "",
            f"> Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            "",
            "---",
            "",
        ]

        # 1. Overview
        lines += self._section_overview(df)

        # 2. Per-modality breakdown
        lines += self._section_per_modality(df)

        # 3. Per-class counts
        lines += self._section_per_class(df)

        # 4. Dataset sources
        lines += self._section_dataset_sources(df)

        # 5. Resolution statistics
        lines += self._section_resolution(df)

        # 6. Extensions
        lines += self._section_extensions(df)

        # 7. Data quality
        lines += self._section_quality(df)

        # 8. Future usage note
        lines += [
            "---",
            "",
            "## Usage by Downstream Modules",
            "",
            "All modules (3–24) should load `metadata/master_metadata.csv` as their",
            "primary data source.  The `internal_id` column provides stable cross-module",
            "references.  Original file paths are preserved in `image_path`.",
            "",
            "```python",
            "import pandas as pd",
            "master = pd.read_csv('metadata/master_metadata.csv')",
            "brain_mri = master[master['modality'] == 'brain_mri']",
            "```",
            "",
        ]

        self._content = "\n".join(lines)
        logger.info("dataset_summary.md generated (%d lines).", len(lines))
        return self._content

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """Save dataset_summary.md to disk."""
        if self._content is None:
            raise RuntimeError("generate() must be called before save().")
        dest_dir = Path(output_dir) if output_dir else self._outputs_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / DATASET_SUMMARY_MD
        dest.write_text(self._content, encoding="utf-8")
        logger.info("dataset_summary.md saved -> %s", dest)
        return dest

    @property
    def content(self) -> Optional[str]:
        """The Markdown content string, or None before generate()."""
        return self._content

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    @staticmethod
    def _section_overview(df: pd.DataFrame) -> list:
        total = len(df)
        valid = int(df["is_valid"].sum()) if "is_valid" in df.columns else total
        invalid = total - valid
        mods = df[COL_MODALITY].nunique() if COL_MODALITY in df.columns else 0
        classes = df[COL_NORMALIZED_CLASS].nunique() if COL_NORMALIZED_CLASS in df.columns else 0
        splits = df[COL_SPLIT].unique().tolist() if COL_SPLIT in df.columns else []
        size_gb = df[COL_FILE_SIZE].sum() / 1e9 if COL_FILE_SIZE in df.columns else 0.0

        return [
            "## Overview",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Total Images | {total:,} |",
            f"| Valid Images | {valid:,} |",
            f"| Invalid Images | {invalid:,} |",
            f"| Total Modalities | {mods} |",
            f"| Total Normalized Classes | {classes} |",
            f"| Dataset Splits | {', '.join(splits)} |",
            f"| Total Dataset Size | {size_gb:.2f} GB |",
            "",
        ]

    @staticmethod
    def _section_per_modality(df: pd.DataFrame) -> list:
        if COL_MODALITY not in df.columns:
            return []
        lines = ["## Images per Modality", "", "| Modality | Total | Valid | Classes | Avg Width | Avg Height |",
                 "|---|---|---|---|---|---|"]
        for mod, grp in sorted(df.groupby(COL_MODALITY)):
            total = len(grp)
            valid = int(grp["is_valid"].sum()) if "is_valid" in grp.columns else total
            classes = grp[COL_NORMALIZED_CLASS].nunique() if COL_NORMALIZED_CLASS in grp.columns else 0
            avg_w = f"{grp[COL_WIDTH].mean():.0f}" if COL_WIDTH in grp.columns else "—"
            avg_h = f"{grp[COL_HEIGHT].mean():.0f}" if COL_HEIGHT in grp.columns else "—"
            lines.append(f"| {mod} | {total:,} | {valid:,} | {classes} | {avg_w} px | {avg_h} px |")
        lines.append("")
        return lines

    @staticmethod
    def _section_per_class(df: pd.DataFrame) -> list:
        if COL_NORMALIZED_CLASS not in df.columns:
            return []
        lines = ["## Images per Disease Class", "", "| Normalized Class | Count | Modalities |",
                 "|---|---|---|"]
        class_counts = df.groupby(COL_NORMALIZED_CLASS).agg(
            count=(COL_NORMALIZED_CLASS, "count"),
            modalities=(COL_MODALITY, lambda x: ", ".join(sorted(x.unique())))
            if COL_MODALITY in df.columns else (COL_NORMALIZED_CLASS, "count")
        ).sort_values("count", ascending=False)
        for cls, row in class_counts.iterrows():
            lines.append(f"| {cls} | {row['count']:,} | {row['modalities']} |")
        lines.append("")
        return lines

    @staticmethod
    def _section_dataset_sources(df: pd.DataFrame) -> list:
        if COL_DATASET_NAME not in df.columns:
            return []
        lines = ["## Dataset Sources", "", "| Dataset Name | Images | Modality |",
                 "|---|---|---|"]
        src_counts = df.groupby(COL_DATASET_NAME).agg(
            count=(COL_DATASET_NAME, "count"),
            modality=(COL_MODALITY, "first") if COL_MODALITY in df.columns else (COL_DATASET_NAME, "first")
        ).sort_values("count", ascending=False)
        for name, row in src_counts.iterrows():
            lines.append(f"| {name} | {row['count']:,} | {row['modality']} |")

        # Largest and smallest dataset
        if not src_counts.empty:
            largest = src_counts["count"].idxmax()
            smallest = src_counts["count"].idxmin()
            lines += [
                "",
                f"**Largest dataset:** {largest} ({src_counts.loc[largest, 'count']:,} images)",
                f"**Smallest dataset:** {smallest} ({src_counts.loc[smallest, 'count']:,} images)",
            ]
        lines.append("")
        return lines

    @staticmethod
    def _section_resolution(df: pd.DataFrame) -> list:
        if COL_WIDTH not in df.columns or COL_HEIGHT not in df.columns:
            return []
        valid_df = df[(df[COL_WIDTH] > 0) & (df[COL_HEIGHT] > 0)]
        lines = [
            "## Image Resolution Statistics",
            "",
            "| Metric | Width (px) | Height (px) |",
            "|---|---|---|",
            f"| Min | {valid_df[COL_WIDTH].min()} | {valid_df[COL_HEIGHT].min()} |",
            f"| Max | {valid_df[COL_WIDTH].max()} | {valid_df[COL_HEIGHT].max()} |",
            f"| Mean | {valid_df[COL_WIDTH].mean():.1f} | {valid_df[COL_HEIGHT].mean():.1f} |",
            f"| Median | {valid_df[COL_WIDTH].median():.0f} | {valid_df[COL_HEIGHT].median():.0f} |",
            "",
        ]
        return lines

    @staticmethod
    def _section_extensions(df: pd.DataFrame) -> list:
        if COL_EXTENSION not in df.columns:
            return []
        ext_counts = df[COL_EXTENSION].str.lower().value_counts()
        lines = ["## Supported File Extensions", "", "| Extension | Count |", "|---|---|"]
        for ext, cnt in ext_counts.items():
            lines.append(f"| {ext} | {cnt:,} |")
        lines.append("")
        return lines

    @staticmethod
    def _section_quality(df: pd.DataFrame) -> list:
        if "is_valid" not in df.columns:
            return []
        valid = int(df["is_valid"].sum())
        invalid = len(df) - valid
        pct = valid / len(df) * 100 if len(df) > 0 else 0.0
        lines = [
            "## Data Quality",
            "",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Valid Images | {valid:,} ({pct:.1f}%) |",
            f"| Invalid / Corrupt Images | {invalid:,} ({100 - pct:.1f}%) |",
            "",
        ]
        return lines
