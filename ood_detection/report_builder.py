"""
ood_detection/report_builder.py
===============================
Report and visualization generator for Phase 8 OOD and Anomaly Detection.
Produces:
  1. outputs/v2_phase8/reports/ood_method_comparison.png
  2. outputs/v2_phase8/reports/ood_score_distributions.png
  3. outputs/v2_phase8/reports/ood_confusion_matrices/*.png
  4. V2.0_PHASE8_OOD_ANOMALY_DETECTION_REPORT.md
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ood_detection.config import (
    ALL_MODALITIES,
    CONF_MAT_DIR,
    METRICS_DIR,
    OOD_MODELS_DIR,
    OUTPUTS_P8_DIR,
    PLOTS_DIR,
    READINESS_DIR,
    REPORTS_DIR,
    ROOT,
    SCORES_DIR,
    THRESHOLDS_DIR,
)

logger = logging.getLogger(__name__)


def generate_method_comparison_plot(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates bar chart comparing AUROC and FPR95 across methods."""
    plt.figure(figsize=(12, 6))

    # Pivot to get mean AUROC per method and modality in Scenario A (cross-modality)
    scen_a = metrics_df[metrics_df["scenario"] == "scenario_a"]
    if scen_a.empty:
        scen_a = metrics_df

    pivot = scen_a.pivot_table(index="modality", columns="method", values="auroc", aggfunc="mean")

    ax = pivot.plot(kind="bar", figsize=(12, 6), colormap="viridis", width=0.8, edgecolor="black")
    plt.title("OOD Detection Performance Across Methods (Mean AUROC — Scenario A)", fontsize=13, fontweight="bold")
    plt.ylabel("AUROC (Higher is Better)", fontsize=11)
    plt.xlabel("Target Modality", fontsize=11)
    plt.ylim(0.4, 1.05)
    plt.axhline(0.5, color="red", linestyle="--", alpha=0.6, label="Random Guess (0.50)")
    plt.grid(True, linestyle=":", alpha=0.5, axis="y")
    plt.legend(title="OOD Method", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved method comparison plot to {output_path}")


def generate_score_distributions_plot(
    scores_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generates score distribution histograms for ID vs OOD across methods."""
    methods = ["msp", "energy", "mahalanobis", "knn"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for idx, method in enumerate(methods):
        ax = axes[idx]
        sub = scores_df[scores_df["method"] == method]
        if sub.empty:
            continue

        id_scores = sub[sub["is_ood"] == 0]["score"].values
        ood_scores = sub[sub["is_ood"] == 1]["score"].values

        # Remove non-finite
        id_scores = id_scores[np.isfinite(id_scores)]
        ood_scores = ood_scores[np.isfinite(ood_scores)]

        if len(id_scores) > 0 and len(ood_scores) > 0:
            # Use 1st and 99th percentiles for clean plotting range
            lo = min(np.percentile(id_scores, 1), np.percentile(ood_scores, 1))
            hi = max(np.percentile(id_scores, 99), np.percentile(ood_scores, 99))
            bins = np.linspace(lo, hi, 35)

            ax.hist(id_scores, bins=bins, alpha=0.6, color="#2980b9", density=True, label="In-Distribution (ID)")
            ax.hist(ood_scores, bins=bins, alpha=0.6, color="#e74c3c", density=True, label="Out-of-Distribution (OOD)")

        ax.set_title(f"Method: {method.upper()} Score Distribution", fontsize=11, fontweight="bold")
        ax.set_xlabel("OOD Score (Higher = More OOD)", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

    plt.suptitle("In-Distribution vs Out-of-Distribution Score Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"Saved score distribution plot to {output_path}")


def generate_confusion_matrix_plot(
    conf_mat: List[List[int]],
    modality: str,
    method: str,
    scenario: str,
    output_path: Path,
) -> None:
    """Plots and saves an individual confusion matrix."""
    tn, fp = conf_mat[0]
    fn, tp = conf_mat[1]

    mat = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.5, 4))

    cax = ax.matshow(mat, cmap="Blues", alpha=0.7)
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            ax.text(j, i, f"{val:,}", ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred ID", "Pred OOD"], fontsize=9)
    ax.set_yticklabels(["True ID", "True OOD"], fontsize=9)
    ax.set_title(f"{modality.upper()} — {method.upper()} ({scenario})", fontsize=10, fontweight="bold", pad=12)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
