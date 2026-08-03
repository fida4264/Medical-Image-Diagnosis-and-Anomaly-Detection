"""
metadata_manager/visualization.py
====================================
Generates 5 publication-quality charts from the standardised master_metadata.

Charts Produced
---------------
1. modality_distribution_pie.png  — pie chart, images per modality
2. class_distribution_bar.png     — horizontal bar, images per normalized class
3. modality_class_heatmap.png     — heatmap of class×modality image counts
4. resolution_scatter.png         — scatter of width vs height (sampled)
5. dataset_source_bar.png         — bar chart, images per dataset source

All charts are saved under metadata/visualizations/.

Classes
-------
Visualizer
    Renders and saves all 5 charts.

Author: Medical AI System — Module 2
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")          # Non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from metadata_manager.config import (
    VISUALIZATIONS_DIR,
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_DATASET_NAME,
    COL_WIDTH,
    COL_HEIGHT,
)

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]
BG_COLOR = "#1A1A2E"
GRID_COLOR = "#2D2D4E"
TEXT_COLOR = "#E0E0E0"
ACCENT_COLOR = "#4C72B0"

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": BG_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "legend.facecolor": "#222240",
    "legend.edgecolor": GRID_COLOR,
    "grid.color": GRID_COLOR,
    "grid.alpha": 0.4,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
})


class Visualizer:
    """
    Generates and saves all 5 visualisation charts.

    Parameters
    ----------
    master_df : pd.DataFrame
        The standardised master_metadata DataFrame.
    output_dir : Path, optional
        Where to save the PNG files.  Defaults to VISUALIZATIONS_DIR.
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> None:
        self._df: pd.DataFrame = master_df
        self._out_dir: Path = Path(output_dir) if output_dir else VISUALIZATIONS_DIR
        self._out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_all(self) -> list:
        """
        Generate all 5 charts and save them to disk.

        Returns
        -------
        list of Path
            Paths of all saved PNG files.
        """
        logger.info("Generating visualisations -> %s", self._out_dir)
        saved = []
        fns = [
            self.plot_modality_pie,
            self.plot_class_distribution,
            self.plot_modality_class_heatmap,
            self.plot_resolution_scatter,
            self.plot_dataset_source_bar,
        ]
        for fn in fns:
            try:
                path = fn()
                saved.append(path)
                logger.info("  Saved: %s", path.name)
            except Exception as exc:
                logger.warning("  Chart failed (%s): %s", fn.__name__, exc)
        logger.info("Visualisations complete. %d charts saved.", len(saved))
        return saved

    def plot_modality_pie(self) -> Path:
        """Pie chart — images per modality."""
        counts = self._df[COL_MODALITY].value_counts()
        fig, ax = plt.subplots(figsize=(9, 7))
        fig.patch.set_facecolor(BG_COLOR)
        wedges, texts, autotexts = ax.pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            colors=PALETTE[: len(counts)],
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
        )
        for t in texts:
            t.set_color(TEXT_COLOR)
            t.set_fontsize(10)
        for at in autotexts:
            at.set_color("#FFFFFF")
            at.set_fontsize(9)
        ax.set_title("Modality Distribution\n(Images per Modality)", pad=18)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        dest = self._out_dir / "modality_distribution_pie.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        return dest

    def plot_class_distribution(self) -> Path:
        """Horizontal bar chart — images per normalized class."""
        if COL_NORMALIZED_CLASS not in self._df.columns:
            raise ValueError("normalized_class column missing")
        counts = (
            self._df[COL_NORMALIZED_CLASS]
            .value_counts()
            .head(30)
            .sort_values()
        )
        fig, ax = plt.subplots(figsize=(12, max(6, len(counts) * 0.4)))
        fig.patch.set_facecolor(BG_COLOR)
        bars = ax.barh(
            counts.index, counts.values,
            color=PALETTE[0], alpha=0.85, edgecolor=BG_COLOR, linewidth=0.5,
        )
        ax.bar_label(bars, fmt="%,.0f", padding=4, color=TEXT_COLOR, fontsize=9)
        ax.set_xlabel("Number of Images")
        ax.set_title("Disease Class Distribution (Top 30)")
        ax.grid(axis="x", linestyle="--")
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        dest = self._out_dir / "class_distribution_bar.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        return dest

    def plot_modality_class_heatmap(self) -> Path:
        """Heatmap — class × modality image count matrix."""
        if COL_NORMALIZED_CLASS not in self._df.columns:
            raise ValueError("normalized_class column missing")
        pivot = (
            self._df
            .groupby([COL_NORMALIZED_CLASS, COL_MODALITY])
            .size()
            .unstack(fill_value=0)
        )
        # Drop 'unknown' to keep chart clean
        pivot = pivot.drop(index="unknown", errors="ignore")

        fig, ax = plt.subplots(figsize=(max(8, pivot.shape[1] * 1.5), max(6, pivot.shape[0] * 0.5)))
        fig.patch.set_facecolor(BG_COLOR)

        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=9)
        ax.set_title("Class × Modality Distribution Heatmap")
        ax.set_facecolor(BG_COLOR)

        # Annotate cells
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:,}", ha="center", va="center",
                            fontsize=7, color="white" if val > pivot.values.max() * 0.5 else TEXT_COLOR)

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
        cbar.ax.tick_params(labelcolor=TEXT_COLOR)
        plt.tight_layout()
        dest = self._out_dir / "modality_class_heatmap.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        return dest

    def plot_resolution_scatter(self) -> Path:
        """Scatter plot — image width vs height (sampled, coloured by modality)."""
        df = self._df
        if COL_WIDTH not in df.columns or COL_HEIGHT not in df.columns:
            raise ValueError("width/height columns missing")
        valid = df[(df[COL_WIDTH] > 0) & (df[COL_HEIGHT] > 0)]
        # Sample at most 5 000 points for readability
        sample = valid.sample(min(5000, len(valid)), random_state=42)

        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(BG_COLOR)
        modalities = sorted(sample[COL_MODALITY].unique())
        for i, mod in enumerate(modalities):
            sub = sample[sample[COL_MODALITY] == mod]
            ax.scatter(
                sub[COL_WIDTH], sub[COL_HEIGHT],
                label=mod, alpha=0.45, s=12,
                color=PALETTE[i % len(PALETTE)],
            )
        ax.set_xlabel("Width (px)")
        ax.set_ylabel("Height (px)")
        ax.set_title("Image Resolution Distribution\n(Sample of 5,000 images)")
        ax.legend(loc="upper right", fontsize=9, framealpha=0.6)
        ax.grid(linestyle="--")
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        dest = self._out_dir / "resolution_scatter.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        return dest

    def plot_dataset_source_bar(self) -> Path:
        """Bar chart — images per dataset source."""
        if COL_DATASET_NAME not in self._df.columns:
            raise ValueError("dataset_name column missing")
        counts = self._df[COL_DATASET_NAME].value_counts().sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(12, max(5, len(counts) * 0.6)))
        fig.patch.set_facecolor(BG_COLOR)
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(counts))]
        bars = ax.barh(
            [str(n)[:45] for n in counts.index],
            counts.values,
            color=colors, alpha=0.85, edgecolor=BG_COLOR,
        )
        ax.bar_label(bars, fmt="%,.0f", padding=4, color=TEXT_COLOR, fontsize=9)
        ax.set_xlabel("Number of Images")
        ax.set_title("Images per Dataset Source")
        ax.grid(axis="x", linestyle="--")
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        dest = self._out_dir / "dataset_source_bar.png"
        plt.savefig(dest, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
        plt.close()
        return dest
