"""
eda/visualizations/charts.py
==============================
Publication-quality chart generation for the EDA module.

All charts use the dark theme (matching Module 2 visualizations).
Every method saves a PNG at 300 DPI and returns the saved Path.

Charts Available
----------------
pie()                — pie chart
bar_h()              — horizontal bar chart
bar_v()              — vertical bar chart
histogram()          — histogram with KDE line
scatter()            — scatter plot (optional colour by category)
violin()             — violin plot per category
box()                — box plot per category
treemap()            — treemap (squarify-based)
line()               — line chart
stacked_bar()        — stacked bar chart
grouped_bar()        — grouped bar chart

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------

BG_COLOR   = "#1A1A2E"
GRID_COLOR = "#2D2D4E"
TEXT_COLOR = "#E0E0E0"
ACCENT     = "#4C72B0"

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]

_BASE_RCPARAMS = {
    "figure.facecolor":  BG_COLOR,
    "axes.facecolor":    BG_COLOR,
    "axes.edgecolor":    GRID_COLOR,
    "axes.labelcolor":   TEXT_COLOR,
    "xtick.color":       TEXT_COLOR,
    "ytick.color":       TEXT_COLOR,
    "text.color":        TEXT_COLOR,
    "legend.facecolor":  "#222240",
    "legend.edgecolor":  GRID_COLOR,
    "grid.color":        GRID_COLOR,
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "axes.labelsize":    11,
}

plt.rcParams.update(_BASE_RCPARAMS)

DPI = 300


def _save(fig: plt.Figure, dest: Path) -> Path:
    """Save figure and close."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("  Saved chart: %s", dest.name)
    return dest


def _color_cycle(n: int) -> List[str]:
    """Return n colours cycling through the palette."""
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


class ChartGenerator:
    """
    Generates and saves publication-quality charts.

    Parameters
    ----------
    output_dir : Path
        Directory where PNG files are saved.
    dpi : int
        Output DPI.  Default 300.
    """

    def __init__(self, output_dir: Path, dpi: int = DPI) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        global DPI
        DPI = dpi

    # ------------------------------------------------------------------
    # Pie chart
    # ------------------------------------------------------------------

    def pie(
        self,
        counts: Dict[str, int],
        title: str,
        filename: str,
        max_slices: int = 10,
    ) -> Path:
        """
        Pie chart for a category → count mapping.

        Parameters
        ----------
        counts : dict
            {label: count}
        title : str
            Chart title.
        filename : str
            Output file name (without extension).
        max_slices : int
            Groups remaining slices into 'Other'.
        """
        labels = list(counts.keys())
        values = list(counts.values())

        # Merge small slices into "Other"
        if len(labels) > max_slices:
            top_labels = labels[:max_slices]
            top_vals   = values[:max_slices]
            other_val  = sum(values[max_slices:])
            if other_val > 0:
                top_labels.append("Other")
                top_vals.append(other_val)
            labels, values = top_labels, top_vals

        colors = _color_cycle(len(labels))
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor(BG_COLOR)

        wedges, texts, autotexts = ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            colors=colors,
            startangle=140,
            pctdistance=0.82,
            wedgeprops={"edgecolor": BG_COLOR, "linewidth": 2},
        )
        for t in texts:
            t.set_color(TEXT_COLOR)
            t.set_fontsize(9)
        for at in autotexts:
            at.set_color("#FFFFFF")
            at.set_fontsize(8)
        ax.set_title(title, pad=18, color=TEXT_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Horizontal bar chart
    # ------------------------------------------------------------------

    def bar_h(
        self,
        counts: Dict[str, int],
        title: str,
        filename: str,
        xlabel: str = "Count",
        max_items: int = 30,
        color: Optional[str] = None,
        show_values: bool = True,
    ) -> Path:
        """Horizontal bar chart."""
        items = sorted(counts.items(), key=lambda x: x[1])[-max_items:]
        labels = [str(k)[:45] for k, _ in items]
        values = [v for _, v in items]
        colors = _color_cycle(len(labels)) if color is None else [color] * len(labels)

        fig_h = max(6, len(labels) * 0.45)
        fig, ax = plt.subplots(figsize=(14, fig_h))
        fig.patch.set_facecolor(BG_COLOR)
        bars = ax.barh(labels, values, color=colors, alpha=0.88, edgecolor=BG_COLOR, linewidth=0.5)
        if show_values:
            ax.bar_label(bars, fmt="%,.0f", padding=5, color=TEXT_COLOR, fontsize=9)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(axis="x", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Vertical bar chart
    # ------------------------------------------------------------------

    def bar_v(
        self,
        counts: Dict[str, int],
        title: str,
        filename: str,
        ylabel: str = "Count",
        color: Optional[str] = None,
        rotate_labels: int = 45,
    ) -> Path:
        """Vertical bar chart."""
        labels = list(counts.keys())
        values = list(counts.values())
        colors = _color_cycle(len(labels)) if color is None else [color] * len(labels)

        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 1.2), 7))
        fig.patch.set_facecolor(BG_COLOR)
        bars = ax.bar(range(len(labels)), values, color=colors, alpha=0.88, edgecolor=BG_COLOR)
        ax.bar_label(bars, fmt="%,.0f", padding=4, color=TEXT_COLOR, fontsize=9)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=rotate_labels, ha="right", fontsize=9, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Histogram
    # ------------------------------------------------------------------

    def histogram(
        self,
        values: Sequence[float],
        title: str,
        filename: str,
        xlabel: str = "Value",
        bins: int = 50,
        color: str = ACCENT,
        show_kde: bool = True,
    ) -> Path:
        """Histogram with optional KDE overlay."""
        arr = np.array(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            logger.warning("histogram: empty array for %s", filename)
            return self._out / f"{filename}.png"

        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(BG_COLOR)
        ax.hist(arr, bins=bins, color=color, alpha=0.75, edgecolor=BG_COLOR, linewidth=0.5)

        if show_kde and len(arr) > 10:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                xs = np.linspace(arr.min(), arr.max(), 300)
                # Scale KDE to match histogram
                bin_width = (arr.max() - arr.min()) / bins if bins > 0 else 1
                ax2 = ax.twinx()
                ax2.plot(xs, kde(xs), color="#FFD700", linewidth=2, label="KDE")
                ax2.set_ylabel("Density", color=TEXT_COLOR)
                ax2.tick_params(colors=TEXT_COLOR)
                ax2.set_facecolor(BG_COLOR)
                ax2.spines["right"].set_color(GRID_COLOR)
                ax2.spines["top"].set_color(GRID_COLOR)
            except ImportError:
                pass

        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel("Count", color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Scatter plot
    # ------------------------------------------------------------------

    def scatter(
        self,
        x: Sequence[float],
        y: Sequence[float],
        title: str,
        filename: str,
        xlabel: str = "X",
        ylabel: str = "Y",
        labels: Optional[Sequence[str]] = None,
        alpha: float = 0.4,
        s: float = 15,
    ) -> Path:
        """Scatter plot, optionally coloured by category label."""
        x_arr = np.array(x, dtype=float)
        y_arr = np.array(y, dtype=float)

        fig, ax = plt.subplots(figsize=(12, 9))
        fig.patch.set_facecolor(BG_COLOR)

        if labels is not None:
            unique_labels = sorted(set(labels))
            color_map = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(unique_labels)}
            for lab in unique_labels:
                mask = np.array([l == lab for l in labels])
                ax.scatter(x_arr[mask], y_arr[mask], label=lab, alpha=alpha,
                           s=s, color=color_map[lab])
            ax.legend(loc="upper right", fontsize=9, framealpha=0.5)
        else:
            ax.scatter(x_arr, y_arr, color=ACCENT, alpha=alpha, s=s)

        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(linestyle="--", alpha=0.3, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Violin plot
    # ------------------------------------------------------------------

    def violin(
        self,
        data_by_category: Dict[str, Sequence[float]],
        title: str,
        filename: str,
        xlabel: str = "Category",
        ylabel: str = "Value",
        rotate_labels: int = 30,
    ) -> Path:
        """Violin plot for multiple categories."""
        categories = list(data_by_category.keys())
        arrays = [np.array(v, dtype=float) for v in data_by_category.values()]
        # Filter empty
        non_empty = [(c, a) for c, a in zip(categories, arrays) if len(a) > 0]
        if not non_empty:
            logger.warning("violin: no data for %s", filename)
            return self._out / f"{filename}.png"
        categories, arrays = zip(*non_empty)

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.5), 8))
        fig.patch.set_facecolor(BG_COLOR)
        parts = ax.violinplot(arrays, positions=range(len(categories)),
                              showmedians=True, showextrema=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(PALETTE[i % len(PALETTE)])
            pc.set_alpha(0.75)
        for part in ("cmedians", "cbars", "cmins", "cmaxes"):
            if part in parts:
                parts[part].set_color(TEXT_COLOR)
                parts[part].set_linewidth(1.5)

        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, rotation=rotate_labels, ha="right",
                           fontsize=9, color=TEXT_COLOR)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Box plot
    # ------------------------------------------------------------------

    def box(
        self,
        data_by_category: Dict[str, Sequence[float]],
        title: str,
        filename: str,
        xlabel: str = "Category",
        ylabel: str = "Value",
        rotate_labels: int = 30,
    ) -> Path:
        """Box plot for multiple categories."""
        categories = list(data_by_category.keys())
        arrays = [np.array(v, dtype=float) for v in data_by_category.values()]
        non_empty = [(c, a) for c, a in zip(categories, arrays) if len(a) > 0]
        if not non_empty:
            return self._out / f"{filename}.png"
        categories, arrays = zip(*non_empty)

        fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.5), 8))
        fig.patch.set_facecolor(BG_COLOR)
        bp = ax.boxplot(arrays, patch_artist=True, notch=False)
        for i, patch in enumerate(bp["boxes"]):
            patch.set_facecolor(PALETTE[i % len(PALETTE)])
            patch.set_alpha(0.8)
        for element in ("whiskers", "caps", "fliers"):
            for item in bp[element]:
                item.set_color(TEXT_COLOR)
        for median in bp["medians"]:
            median.set_color("#FFD700")
            median.set_linewidth(2)

        ax.set_xticks(range(1, len(categories) + 1))
        ax.set_xticklabels(categories, rotation=rotate_labels, ha="right",
                           fontsize=9, color=TEXT_COLOR)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Treemap
    # ------------------------------------------------------------------

    def treemap(
        self,
        counts: Dict[str, int],
        title: str,
        filename: str,
        max_items: int = 30,
    ) -> Path:
        """Treemap using squarify library (falls back to bar chart if not installed)."""
        try:
            import squarify  # optional dependency
            items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:max_items]
            labels_list = [f"{k}\n{v:,}" for k, v in items]
            vals = [v for _, v in items]
            colors = _color_cycle(len(vals))

            fig, ax = plt.subplots(figsize=(16, 10))
            fig.patch.set_facecolor(BG_COLOR)
            squarify.plot(sizes=vals, label=labels_list, color=colors,
                          alpha=0.85, ax=ax, text_kwargs={"color": "white", "fontsize": 9})
            ax.set_title(title, color=TEXT_COLOR, pad=12)
            ax.axis("off")
            ax.set_facecolor(BG_COLOR)
            plt.tight_layout()
        except ImportError:
            # Fall back to horizontal bar chart
            logger.info("squarify not installed — using bar chart for treemap %s", filename)
            return self.bar_h(counts, title, filename, max_items=max_items)

        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Line chart
    # ------------------------------------------------------------------

    def line(
        self,
        x: Sequence,
        y: Sequence[float],
        title: str,
        filename: str,
        xlabel: str = "X",
        ylabel: str = "Y",
        color: str = ACCENT,
        marker: str = "o",
    ) -> Path:
        """Simple line chart."""
        fig, ax = plt.subplots(figsize=(12, 7))
        fig.patch.set_facecolor(BG_COLOR)
        ax.plot(x, y, color=color, linewidth=2, marker=marker, markersize=5)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.grid(linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Stacked bar chart
    # ------------------------------------------------------------------

    def stacked_bar(
        self,
        data: Dict[str, Dict[str, int]],
        title: str,
        filename: str,
        xlabel: str = "Category",
        ylabel: str = "Count",
        rotate_labels: int = 30,
    ) -> Path:
        """
        Stacked bar chart.

        Parameters
        ----------
        data : dict
            {category: {stack_label: count}}
        """
        categories = list(data.keys())
        stack_labels = sorted({k for v in data.values() for k in v})
        colors = _color_cycle(len(stack_labels))

        fig, ax = plt.subplots(figsize=(max(12, len(categories) * 1.2), 8))
        fig.patch.set_facecolor(BG_COLOR)

        bottoms = np.zeros(len(categories))
        for i, sl in enumerate(stack_labels):
            heights = np.array([data[cat].get(sl, 0) for cat in categories], dtype=float)
            ax.bar(categories, heights, bottom=bottoms, color=colors[i],
                   alpha=0.85, label=sl, edgecolor=BG_COLOR, linewidth=0.5)
            bottoms += heights

        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.tick_params(axis="x", rotation=rotate_labels)
        ax.legend(fontsize=9, framealpha=0.5, facecolor="#222240")
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Grouped bar chart
    # ------------------------------------------------------------------

    def grouped_bar(
        self,
        data: Dict[str, Dict[str, int]],
        title: str,
        filename: str,
        xlabel: str = "Group",
        ylabel: str = "Count",
        rotate_labels: int = 30,
    ) -> Path:
        """
        Grouped bar chart.

        Parameters
        ----------
        data : dict
            {group: {bar_label: count}}
        """
        groups = list(data.keys())
        bar_labels = sorted({k for v in data.values() for k in v})
        n_bars = len(bar_labels)
        x = np.arange(len(groups))
        width = 0.8 / n_bars
        colors = _color_cycle(n_bars)

        fig, ax = plt.subplots(figsize=(max(12, len(groups) * 1.5), 8))
        fig.patch.set_facecolor(BG_COLOR)

        for i, bl in enumerate(bar_labels):
            heights = [data[g].get(bl, 0) for g in groups]
            offset = (i - n_bars / 2 + 0.5) * width
            ax.bar(x + offset, heights, width=width * 0.9, color=colors[i],
                   alpha=0.85, label=bl, edgecolor=BG_COLOR)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=rotate_labels, ha="right", fontsize=9, color=TEXT_COLOR)
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
        ax.set_title(title, color=TEXT_COLOR)
        ax.legend(fontsize=9, framealpha=0.5, facecolor="#222240")
        ax.grid(axis="y", linestyle="--", alpha=0.4, color=GRID_COLOR)
        ax.set_facecolor(BG_COLOR)
        plt.tight_layout()
        return _save(fig, self._out / f"{filename}.png")
