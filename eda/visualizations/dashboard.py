"""
eda/visualizations/dashboard.py
=================================
Multi-panel dashboard figures for the EDA module.

Each dashboard consolidates several charts into a single high-level
summary figure (one PNG per dashboard).

Dashboards Available
--------------------
DashboardGenerator.dataset_overview()   — total images, modality + class bars
DashboardGenerator.quality_dashboard()  — quality score gauge + flag breakdown
DashboardGenerator.augmentation_dashboard() — before/after class balance comparison

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared style constants (must match charts.py)
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

_RCPARAMS = {
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
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "axes.labelsize":    10,
}

plt.rcParams.update(_RCPARAMS)

DPI = 200  # Dashboards use slightly lower DPI than individual charts


def _color_cycle(n: int) -> List[str]:
    return [PALETTE[i % len(PALETTE)] for i in range(n)]


def _save(fig: plt.Figure, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=DPI, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    logger.info("  Saved dashboard: %s", dest.name)
    return dest


def _style_ax(ax: plt.Axes) -> None:
    """Apply consistent dark theme to an axes."""
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(GRID_COLOR)


def _horizontal_bar(
    ax: plt.Axes,
    counts: Dict[str, int],
    title: str,
    max_items: int = 15,
    xlabel: str = "Count",
) -> None:
    """Draw a horizontal bar chart onto *ax*."""
    items = sorted(counts.items(), key=lambda x: x[1])[-max_items:]
    labels = [str(k)[:35] for k, _ in items]
    values = [v for _, v in items]
    colors = _color_cycle(len(labels))

    bars = ax.barh(labels, values, color=colors, alpha=0.88, edgecolor=BG_COLOR)
    ax.bar_label(bars, fmt="%,.0f", padding=4, color=TEXT_COLOR, fontsize=8)
    ax.set_xlabel(xlabel, color=TEXT_COLOR, fontsize=9)
    ax.set_title(title, color=TEXT_COLOR)
    ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_COLOR)
    _style_ax(ax)


def _pie(
    ax: plt.Axes,
    counts: Dict[str, int],
    title: str,
    max_slices: int = 8,
) -> None:
    """Draw a pie chart onto *ax*."""
    labels_list = list(counts.keys())
    values = list(counts.values())

    if len(labels_list) > max_slices:
        top_l = labels_list[:max_slices]
        top_v = values[:max_slices]
        other = sum(values[max_slices:])
        if other > 0:
            top_l.append("Other")
            top_v.append(other)
        labels_list, values = top_l, top_v

    colors = _color_cycle(len(labels_list))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels_list,
        autopct="%1.1f%%",
        colors=colors,
        startangle=140,
        pctdistance=0.80,
        wedgeprops={"edgecolor": BG_COLOR, "linewidth": 1.5},
    )
    for t in texts:
        t.set_color(TEXT_COLOR)
        t.set_fontsize(8)
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(7)
    ax.set_title(title, color=TEXT_COLOR)
    ax.set_facecolor(BG_COLOR)


# ---------------------------------------------------------------------------
# DashboardGenerator
# ---------------------------------------------------------------------------


class DashboardGenerator:
    """
    Generates multi-panel summary dashboards.

    Parameters
    ----------
    output_dir : Path
        Directory where PNG dashboards are saved.
    dpi : int
        Output DPI (default 200).
    """

    def __init__(self, output_dir: Path, dpi: int = DPI) -> None:
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        global DPI
        DPI = dpi

    # ------------------------------------------------------------------
    # Dataset Overview Dashboard
    # ------------------------------------------------------------------

    def dataset_overview(
        self,
        dataset_stats: Dict[str, Any],
        class_stats: Dict[str, Any],
        modality_stats: Dict[str, Any],
        filename: str = "dashboard_dataset_overview",
    ) -> Path:
        """
        4-panel dataset overview dashboard.

        Panels
        ------
        1. KPI cards (total images, modalities, classes, sources)
        2. Images per modality (pie)
        3. Images per modality (horizontal bar)
        4. Top-20 disease classes (horizontal bar)
        """
        fig = plt.figure(figsize=(22, 14), facecolor=BG_COLOR)
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

        # --- Panel 1: KPI cards ---
        ax_kpi = fig.add_subplot(gs[0, 0])
        self._kpi_panel(
            ax=ax_kpi,
            kpis=[
                ("Total Images",     f"{dataset_stats.get('total_images', 0):,}"),
                ("Modalities",       str(dataset_stats.get("total_modalities", 0))),
                ("Disease Classes",  str(dataset_stats.get("total_disease_classes", 0))),
                ("Dataset Sources",  str(dataset_stats.get("total_dataset_sources", 0))),
                ("Balance",          class_stats.get("balance_label", "N/A")),
                ("Gini Coefficient", f"{class_stats.get('gini_coefficient', 0):.3f}"),
            ],
            title="Dataset At a Glance",
        )

        # --- Panel 2: Modality pie ---
        ax_pie = fig.add_subplot(gs[0, 1])
        mod_counts = dataset_stats.get("images_per_modality", {})
        if mod_counts:
            _pie(ax_pie, mod_counts, "Images by Modality", max_slices=8)

        # --- Panel 3: Modality bar ---
        ax_mod_bar = fig.add_subplot(gs[0, 2])
        if mod_counts:
            _horizontal_bar(ax_mod_bar, mod_counts, "Images per Modality")

        # --- Panel 4: Top classes ---
        ax_cls = fig.add_subplot(gs[1, :2])
        top20 = dict(
            sorted(
                class_stats.get("class_counts", {}).items(),
                key=lambda x: x[1],
                reverse=True,
            )[:20]
        )
        if top20:
            _horizontal_bar(ax_cls, top20, "Top 20 Disease Classes by Image Count", max_items=20)

        # --- Panel 5: Split distribution ---
        ax_split = fig.add_subplot(gs[1, 2])
        split_counts = dataset_stats.get("split_distribution", {})
        if split_counts:
            _pie(ax_split, split_counts, "Train / Val / Test Split", max_slices=5)

        fig.suptitle(
            "Medical Dataset — Overview Dashboard",
            fontsize=18, fontweight="bold", color=TEXT_COLOR, y=1.01,
        )
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Quality Dashboard
    # ------------------------------------------------------------------

    def quality_dashboard(
        self,
        quality_stats: Dict[str, Any],
        filename: str = "dashboard_quality",
    ) -> Path:
        """
        4-panel quality dashboard.

        Panels
        ------
        1. Quality score gauge (donut chart)
        2. Quality score component bars
        3. Flag breakdown horizontal bar
        4. Per-modality flag rate bar
        """
        fig = plt.figure(figsize=(20, 12), facecolor=BG_COLOR)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        q_score = quality_stats.get("quality_score", {})
        total_score = q_score.get("total_score", 0.0)
        grade = q_score.get("grade", "N/A")

        # --- Panel 1: Quality gauge (donut) ---
        ax_gauge = fig.add_subplot(gs[0, 0])
        self._quality_gauge(ax_gauge, total_score, grade)

        # --- Panel 2: Score components ---
        ax_components = fig.add_subplot(gs[0, 1])
        components = {
            "Class Balance": q_score.get("class_balance_score", 0),
            "Image Quality": q_score.get("image_quality_score", 0),
            "Corruption": q_score.get("corruption_score", 0),
            "Resolution": q_score.get("resolution_consistency_score", 0),
        }
        self._score_component_bars(ax_components, components, max_per_component=25)

        # --- Panel 3: Flag breakdown ---
        ax_flags = fig.add_subplot(gs[1, 0])
        flag_breakdown = quality_stats.get("flag_breakdown", {})
        flag_pct = {
            k: v.get("pct", 0)
            for k, v in flag_breakdown.items()
            if v.get("pct", 0) > 0
        }
        if flag_pct:
            _horizontal_bar(
                ax_flags, {k: v for k, v in flag_pct.items()},
                "Quality Flag Distribution (%)", xlabel="% of Images",
            )
        else:
            ax_flags.text(
                0.5, 0.5, "No quality flags found",
                ha="center", va="center", color=TEXT_COLOR, fontsize=12,
                transform=ax_flags.transAxes,
            )
            ax_flags.set_title("Quality Flag Distribution", color=TEXT_COLOR)
            _style_ax(ax_flags)

        # --- Panel 4: Per-modality flag rate ---
        ax_mod_flags = fig.add_subplot(gs[1, 1])
        per_mod_q = quality_stats.get("per_modality_quality", {})
        mod_flag_rates = {
            mod: data.get("flag_rate_pct", 0)
            for mod, data in per_mod_q.items()
        }
        if mod_flag_rates:
            _horizontal_bar(ax_mod_flags, mod_flag_rates, "Flag Rate by Modality (%)", xlabel="% Flagged")

        fig.suptitle(
            "Medical Dataset — Quality Dashboard",
            fontsize=18, fontweight="bold", color=TEXT_COLOR, y=1.01,
        )
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Augmentation Dashboard
    # ------------------------------------------------------------------

    def augmentation_dashboard(
        self,
        aug_stats: Dict[str, Any],
        filename: str = "dashboard_augmentation",
    ) -> Path:
        """
        3-panel augmentation dashboard.

        Panels
        ------
        1. Before vs After class balance (top-15 classes, grouped bar)
        2. Transformation frequency (horizontal bar)
        3. Per-modality augmentation factor (vertical bar)
        """
        fig = plt.figure(figsize=(22, 12), facecolor=BG_COLOR)
        gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # --- Panel 1: Before / after class balance ---
        ax_balance = fig.add_subplot(gs[0, :])
        cb = aug_stats.get("class_balance_improvement", {})
        if cb.get("available"):
            before = cb.get("class_counts_before", {})
            after  = cb.get("class_counts_after", {})
            # Top 15 classes by "after" count
            top15 = sorted(after, key=lambda k: after[k], reverse=True)[:15]
            before_vals = [before.get(c, 0) for c in top15]
            after_vals  = [after.get(c, 0) for c in top15]

            x = np.arange(len(top15))
            w = 0.38
            bars1 = ax_balance.bar(x - w / 2, before_vals, w, color=PALETTE[0],
                                   alpha=0.85, label="Before Aug", edgecolor=BG_COLOR)
            bars2 = ax_balance.bar(x + w / 2, after_vals,  w, color=PALETTE[2],
                                   alpha=0.85, label="After Aug",  edgecolor=BG_COLOR)

            ax_balance.set_xticks(x)
            ax_balance.set_xticklabels(top15, rotation=35, ha="right", fontsize=8, color=TEXT_COLOR)
            ax_balance.set_ylabel("Image Count", color=TEXT_COLOR)
            ax_balance.set_title(
                f"Class Balance Before vs After Augmentation  "
                f"(Gini: {cb.get('gini_before', 0):.3f} → {cb.get('gini_after', 0):.3f})",
                color=TEXT_COLOR,
            )
            ax_balance.legend(fontsize=9, framealpha=0.5, facecolor="#222240")
            ax_balance.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
            _style_ax(ax_balance)
        else:
            ax_balance.text(
                0.5, 0.5, "Augmentation data not available",
                ha="center", va="center", color=TEXT_COLOR, fontsize=13,
                transform=ax_balance.transAxes,
            )
            ax_balance.set_title("Class Balance Before vs After Augmentation", color=TEXT_COLOR)
            _style_ax(ax_balance)

        # --- Panel 2: Transformation frequency ---
        ax_trans = fig.add_subplot(gs[1, 0])
        ops_freq = aug_stats.get("transformation_frequency", {})
        if ops_freq:
            top_ops = dict(sorted(ops_freq.items(), key=lambda x: x[1], reverse=True)[:15])
            _horizontal_bar(ax_trans, top_ops, "Top Augmentation Transformations")
        else:
            ax_trans.text(
                0.5, 0.5, "No transformation data",
                ha="center", va="center", color=TEXT_COLOR, fontsize=12,
                transform=ax_trans.transAxes,
            )
            ax_trans.set_title("Transformation Frequency", color=TEXT_COLOR)
            _style_ax(ax_trans)

        # --- Panel 3: Per-modality augmentation factor ---
        ax_factor = fig.add_subplot(gs[1, 1])
        per_mod = aug_stats.get("per_modality", {})
        mod_factors = {
            mod: data.get("avg_aug_factor", 0)
            for mod, data in per_mod.items()
        }
        if mod_factors:
            mods = list(mod_factors.keys())
            vals = [mod_factors[m] for m in mods]
            colors = _color_cycle(len(mods))
            bars = ax_factor.bar(mods, vals, color=colors, alpha=0.88, edgecolor=BG_COLOR)
            ax_factor.bar_label(bars, fmt="%.1f×", padding=4, color=TEXT_COLOR, fontsize=9)
            ax_factor.set_ylabel("Avg Augmentation Factor", color=TEXT_COLOR)
            ax_factor.set_title("Augmentation Factor by Modality", color=TEXT_COLOR)
            ax_factor.tick_params(axis="x", rotation=30)
            ax_factor.grid(axis="y", linestyle="--", alpha=0.35, color=GRID_COLOR)
            _style_ax(ax_factor)
        else:
            ax_factor.text(
                0.5, 0.5, "No per-modality data",
                ha="center", va="center", color=TEXT_COLOR, fontsize=12,
                transform=ax_factor.transAxes,
            )
            ax_factor.set_title("Augmentation Factor by Modality", color=TEXT_COLOR)
            _style_ax(ax_factor)

        fig.suptitle(
            "Medical Dataset — Augmentation Dashboard",
            fontsize=18, fontweight="bold", color=TEXT_COLOR, y=1.01,
        )
        return _save(fig, self._out / f"{filename}.png")

    # ------------------------------------------------------------------
    # Private helper panels
    # ------------------------------------------------------------------

    def _kpi_panel(
        self,
        ax: plt.Axes,
        kpis: List[tuple],
        title: str,
    ) -> None:
        """Render a text-box-style KPI card panel."""
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        ax.set_facecolor(BG_COLOR)
        ax.set_title(title, color=TEXT_COLOR, fontsize=12, fontweight="bold", pad=10)

        n = len(kpis)
        row_h = 1.0 / (n + 0.5)
        for i, (label, value) in enumerate(kpis):
            y = 1.0 - (i + 0.7) * row_h
            # Background card
            rect = mpatches.FancyBboxPatch(
                (0.02, y - 0.07), 0.96, row_h * 0.9,
                boxstyle="round,pad=0.02",
                linewidth=0,
                facecolor="#252545",
                zorder=1,
            )
            ax.add_patch(rect)
            ax.text(0.08, y + 0.05, label, fontsize=9, color="#A0A0C0",
                    va="center", ha="left", zorder=2)
            ax.text(0.92, y + 0.05, value, fontsize=11, color=TEXT_COLOR,
                    va="center", ha="right", fontweight="bold", zorder=2)

    def _quality_gauge(
        self,
        ax: plt.Axes,
        score: float,
        grade: str,
    ) -> None:
        """Render a donut-style quality gauge."""
        ax.set_facecolor(BG_COLOR)
        ax.axis("equal")
        ax.set_title("Dataset Quality Score", color=TEXT_COLOR)

        # Donut: filled vs remaining
        filled = max(0.0, min(100.0, score))
        remaining = 100.0 - filled

        # Color based on grade
        if filled >= 85:
            fill_color = "#55A868"   # green
        elif filled >= 70:
            fill_color = "#CCB974"   # yellow
        elif filled >= 55:
            fill_color = "#DD8452"   # orange
        else:
            fill_color = "#C44E52"   # red

        wedges, _ = ax.pie(
            [filled, remaining],
            colors=[fill_color, GRID_COLOR],
            startangle=90,
            wedgeprops={"width": 0.45, "edgecolor": BG_COLOR, "linewidth": 2},
        )
        ax.text(0, 0.1, f"{score:.1f}", fontsize=28, ha="center", va="center",
                color=TEXT_COLOR, fontweight="bold")
        ax.text(0, -0.22, "/100", fontsize=12, ha="center", va="center", color="#A0A0C0")
        ax.text(0, -0.48, grade, fontsize=10, ha="center", va="center",
                color=fill_color, fontweight="bold")

    def _score_component_bars(
        self,
        ax: plt.Axes,
        components: Dict[str, float],
        max_per_component: float = 25.0,
    ) -> None:
        """Render a horizontal bar chart of score components."""
        labels = list(components.keys())
        values = [min(v, max_per_component) for v in components.values()]
        colors = _color_cycle(len(labels))

        bars = ax.barh(labels, values, color=colors, alpha=0.85,
                       edgecolor=BG_COLOR, linewidth=0.5)
        ax.bar_label(bars, fmt="%.1f", padding=4, color=TEXT_COLOR, fontsize=9)
        ax.set_xlim(0, max_per_component * 1.15)
        ax.axvline(max_per_component, linestyle="--", color=GRID_COLOR, linewidth=1, alpha=0.6)
        ax.set_xlabel(f"Score (max {max_per_component:.0f} each)", color=TEXT_COLOR, fontsize=9)
        ax.set_title("Quality Score Components", color=TEXT_COLOR)
        ax.grid(axis="x", linestyle="--", alpha=0.35, color=GRID_COLOR)
        _style_ax(ax)
