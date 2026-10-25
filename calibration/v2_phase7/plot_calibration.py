"""
calibration/v2_phase7/plot_calibration.py
=========================================
Generates reliability diagrams and calibration plots for Phase 7.
Reads reliability_data.csv and calibration_results.csv from outputs/v2_phase7/calibration/
and produces:
  - {modality}_reliability.png for each multiclass modality
  - calibration_overview.png (multi-panel figure)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CAL_DIR = ROOT / "outputs" / "v2_phase7" / "calibration"
PHASE7_DIR = ROOT / "outputs" / "v2_phase7"


def plot_modality_reliability(
    modality: str,
    df_rel: pd.DataFrame,
    meta: Dict[str, float],
    output_path: Path,
) -> None:
    """Plot reliability diagram for a single modality before and after calibration."""
    df_base = df_rel[(df_rel["modality"] == modality) & (df_rel["calibrated"] == False)]
    df_cal = df_rel[(df_rel["modality"] == modality) & (df_rel["calibrated"] == True)]

    fig, (ax_main, ax_hist) = plt.subplots(
        2, 1, figsize=(6, 6.5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )

    # Diagonal: perfect calibration
    ax_main.plot([0, 1], [0, 1], "k--", alpha=0.7, label="Perfect calibration")

    # Plot baseline
    confs_base = []
    accs_base = []
    for _, row in df_base.iterrows():
        if pd.notna(row["bin_confidence"]) and pd.notna(row["bin_accuracy"]):
            confs_base.append(row["bin_confidence"])
            accs_base.append(row["bin_accuracy"])

    if confs_base:
        ax_main.plot(
            confs_base,
            accs_base,
            marker="o",
            color="#e74c3c",
            linewidth=2,
            label=f"Baseline (ECE: {meta.get('val_ece_before', 0.0):.4f})",
        )

    # Plot calibrated
    confs_cal = []
    accs_cal = []
    for _, row in df_cal.iterrows():
        if pd.notna(row["bin_confidence"]) and pd.notna(row["bin_accuracy"]):
            confs_cal.append(row["bin_confidence"])
            accs_cal.append(row["bin_accuracy"])

    if confs_cal:
        ax_main.plot(
            confs_cal,
            accs_cal,
            marker="s",
            color="#2980b9",
            linewidth=2,
            label=f"Calibrated (ECE: {meta.get('val_ece_after', 0.0):.4f}, T={meta.get('temperature', 1.0):.3f})",
        )

    title_mod = modality.replace("_", " ").title()
    ax_main.set_title(f"{title_mod} — Reliability Diagram", fontsize=12, fontweight="bold")
    ax_main.set_ylabel("Empirical Accuracy", fontsize=10)
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1.05)
    ax_main.grid(True, linestyle=":", alpha=0.6)
    ax_main.legend(loc="upper left", fontsize=9)

    # Histogram of bin counts
    bin_centers = []
    counts_base = []
    counts_cal = []
    for _, row in df_base.iterrows():
        mid = (row["bin_lower"] + row["bin_upper"]) / 2.0
        bin_centers.append(mid)
        counts_base.append(row["bin_count"])

    for _, row in df_cal.iterrows():
        counts_cal.append(row["bin_count"])

    width = 0.03
    ax_hist.bar(
        [c - width / 2 for c in bin_centers],
        counts_base,
        width=width,
        color="#e74c3c",
        alpha=0.6,
        label="Baseline counts",
    )
    ax_hist.bar(
        [c + width / 2 for c in bin_centers],
        counts_cal,
        width=width,
        color="#2980b9",
        alpha=0.6,
        label="Calibrated counts",
    )
    ax_hist.set_xlabel("Confidence", fontsize=10)
    ax_hist.set_ylabel("Count", fontsize=10)
    ax_hist.grid(True, linestyle=":", alpha=0.4)
    ax_hist.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_overview_grid(
    modalities: List[str],
    df_rel: pd.DataFrame,
    meta_dict: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Generate 2x3 overview plot for all multiclass modalities."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, mod in enumerate(modalities):
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Perfect")

        df_base = df_rel[(df_rel["modality"] == mod) & (df_rel["calibrated"] == False)]
        df_cal = df_rel[(df_rel["modality"] == mod) & (df_rel["calibrated"] == True)]
        meta = meta_dict.get(mod, {})

        confs_base = [r["bin_confidence"] for _, r in df_base.iterrows() if pd.notna(r["bin_confidence"])]
        accs_base = [r["bin_accuracy"] for _, r in df_base.iterrows() if pd.notna(r["bin_accuracy"])]
        if confs_base:
            ax.plot(
                confs_base,
                accs_base,
                marker="o",
                color="#e74c3c",
                linewidth=1.8,
                label=f"Base (ECE: {meta.get('val_ece_before', 0.0):.4f})",
            )

        confs_cal = [r["bin_confidence"] for _, r in df_cal.iterrows() if pd.notna(r["bin_confidence"])]
        accs_cal = [r["bin_accuracy"] for _, r in df_cal.iterrows() if pd.notna(r["bin_accuracy"])]
        if confs_cal:
            ax.plot(
                confs_cal,
                accs_cal,
                marker="s",
                color="#2980b9",
                linewidth=1.8,
                label=f"Cal (ECE: {meta.get('val_ece_after', 0.0):.4f}, T={meta.get('temperature', 1.0):.2f})",
            )

        title_mod = mod.replace("_", " ").title()
        ax.set_title(title_mod, fontsize=11, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("Confidence", fontsize=9)
        ax.set_ylabel("Accuracy", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(loc="upper left", fontsize=8)

    plt.suptitle("Phase 7 — Multiclass Confidence Calibration (Reliability Diagrams)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    rel_csv = CAL_DIR / "reliability_data.csv"
    res_csv = CAL_DIR / "calibration_results.csv"

    if not rel_csv.exists() or not res_csv.exists():
        print(f"Required files not found in {CAL_DIR}")
        return

    df_rel = pd.read_csv(rel_csv)
    df_res = pd.read_csv(res_csv)

    meta_dict = {}
    for _, row in df_res.iterrows():
        meta_dict[row["modality"]] = dict(row)

    modalities = ["brain_mri", "brain_ct", "chest_xray", "histopathology", "skin", "ultrasound"]

    for mod in modalities:
        out_p = CAL_DIR / f"{mod}_calibration.png"
        plot_modality_reliability(mod, df_rel, meta_dict.get(mod, {}), out_p)

    overview_p = CAL_DIR / "calibration_overview.png"
    plot_overview_grid(modalities, df_rel, meta_dict, overview_p)


if __name__ == "__main__":
    main()
