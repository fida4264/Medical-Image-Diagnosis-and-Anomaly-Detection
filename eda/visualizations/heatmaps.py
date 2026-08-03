"""
eda/visualizations/heatmaps.py
================================
Heatmap generation for the EDA module.

Charts Available
----------------
correlation_heatmap()        — numeric correlation matrix
class_modality_heatmap()     — class counts per modality (annotated)
resolution_density_heatmap() — 2D density of width vs height

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BG_COLOR   = "#1A1A2E"
TEXT_COLOR = "#E0E0E0"
GRID_COLOR = "#2D2D4E"
DPI = 300


def _save(fig: plt.Figure, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("  Saved heatmap: %s", dest.name)
    return dest


class HeatmapGenerator:
    """
    Generates heatmap figures.

    Parameters
    ----------
    output_dir : Path
        Directory where PNG files are saved.
    dpi : int
        Output DPI.
    """

    def __init__(self, output_dir: Path, dpi: int = DPI) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        global DPI
        DPI = dpi

    # ------------------------------------------------------------------
    # Correlation heatmap
    # ------------------------------------------------------------------

    def correlation_heatmap(
        self,
        df: pd.DataFrame,
        title: str,
        filename: str,
        columns: Optional[Sequence[str]] = None,
        method: str = "pearson",
        cmap: str = "coolwarm",
    ) -> Path:
        """
        Pearson / Spearman correlation matrix heatmap.

        Parameters
        ----------
        df : DataFrame
            Source data.
        columns : list of str, optional
            Columns to include. Defaults to all numeric columns.
        method : str
            'pearson' or 'spearman'.
        """
        if columns:
            df = df[[c for c in columns if c in df.columns]]
        numeric = df.select_dtypes(include=[np.number])
        if numeric.empty or len(numeric.columns) < 2:
            logger.warning("correlation_heatmap: insufficient numeric columns for %s", filename)
            return self._out / f"{filename}.png"

        corr = numeric.corr(method=method)
        n = len(corr)

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), max(6, n * 0.8)))
        fig.patch.set_facecolor(BG_COLOR)

        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.set_label("Correlation", color=TEXT_COLOR)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9, color=TEXT_COLOR)
        ax.set_yticklabels(corr.index, fontsize=9, color=TEXT_COLOR)

        # Annotate cells
        for i in range(n):
            for j in range(n):
                val = corr.iloc[i, j]
                text_c = "white" if abs(val) > 0.6 else TEXT_COLOR
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=8, color=text_c)

        ax.set_title(title, color=TEXT_COLOR, pad=12)
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Class × Modality heatmap
    # ------------------------------------------------------------------

    def class_modality_heatmap(
        self,
        df: pd.DataFrame,
        class_col: str,
        modality_col: str,
        title: str,
        filename: str,
        top_n_classes: int = 20,
        cmap: str = "YlOrRd",
        log_scale: bool = True,
    ) -> Path:
        """
        Heatmap of image counts per (class × modality).

        Parameters
        ----------
        df : DataFrame
            master_metadata DataFrame.
        class_col : str
            Column for class labels.
        modality_col : str
            Column for modality labels.
        top_n_classes : int
            Show only top-N classes by total count.
        log_scale : bool
            Apply log1p scaling to the colour axis.
        """
        if class_col not in df.columns or modality_col not in df.columns:
            logger.warning("class_modality_heatmap: required columns not found for %s", filename)
            return self._out / f"{filename}.png"

        # Pivot table
        pivot = df.groupby([class_col, modality_col]).size().unstack(fill_value=0)

        # Limit to top-N classes
        top_classes = pivot.sum(axis=1).nlargest(top_n_classes).index
        pivot = pivot.loc[top_classes]

        matrix = pivot.values.astype(float)
        if log_scale:
            matrix = np.log1p(matrix)

        n_rows, n_cols = matrix.shape
        fig_h = max(8, n_rows * 0.45)
        fig_w = max(8, n_cols * 1.2)
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        fig.patch.set_facecolor(BG_COLOR)

        im = ax.imshow(matrix, cmap=cmap, aspect="auto")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        label_suffix = " (log1p)" if log_scale else ""
        cbar.set_label(f"Count{label_suffix}", color=TEXT_COLOR)

        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(n_rows))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=10, color=TEXT_COLOR)
        ax.set_yticklabels(pivot.index, fontsize=9, color=TEXT_COLOR)

        # Annotate with raw counts
        raw = pivot.values
        for i in range(n_rows):
            for j in range(n_cols):
                val = raw[i, j]
                if val > 0:
                    ax.text(j, i, f"{val:,}", ha="center", va="center",
                            fontsize=7, color="white" if matrix[i, j] > matrix.max() * 0.5 else TEXT_COLOR)

        ax.set_xlabel("Modality", color=TEXT_COLOR)
        ax.set_ylabel("Disease Class", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR, pad=12)
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Resolution density heatmap (2D histogram)
    # ------------------------------------------------------------------

    def resolution_density_heatmap(
        self,
        widths: Sequence[float],
        heights: Sequence[float],
        title: str,
        filename: str,
        bins: int = 50,
        cmap: str = "inferno",
    ) -> Path:
        """
        2D density heatmap of image widths vs heights.

        Parameters
        ----------
        widths : sequence of float
        heights : sequence of float
        bins : int
            Number of 2D bins per axis.
        """
        w = np.array(widths, dtype=float)
        h = np.array(heights, dtype=float)
        valid = np.isfinite(w) & np.isfinite(h) & (w > 0) & (h > 0)
        w, h = w[valid], h[valid]

        if len(w) < 2:
            logger.warning("resolution_density_heatmap: insufficient data for %s", filename)
            return self._out / f"{filename}.png"

        fig, ax = plt.subplots(figsize=(10, 9))
        fig.patch.set_facecolor(BG_COLOR)

        h2d, xedges, yedges = np.histogram2d(w, h, bins=bins)
        ax.pcolormesh(xedges, yedges, h2d.T, cmap=cmap)
        im = ax.pcolormesh(xedges, yedges, h2d.T, cmap=cmap)
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(colors=TEXT_COLOR)
        cbar.set_label("Count", color=TEXT_COLOR)

        ax.set_xlabel("Width (pixels)", color=TEXT_COLOR)
        ax.set_ylabel("Height (pixels)", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR, pad=12)
        ax.set_facecolor(BG_COLOR)
        for spine in ax.spines.values():
            spine.set_color(GRID_COLOR)

        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")
