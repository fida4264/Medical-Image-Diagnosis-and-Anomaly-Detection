"""
chest_xray/evaluation/confusion_matrix.py
==========================================
Standalone confusion matrix helper for Chest X-ray evaluation.
New in Module 8 — available to future modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str | Path,
    title: str = "Confusion Matrix",
) -> Path:
    """
    Plots raw and normalized confusion matrix side-by-side and saves to PNG.
    """
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[0], cbar=False,
    )
    axes[0].set_title(f"{title} (Counts)")
    axes[0].set_xlabel("Predicted Disease")
    axes[0].set_ylabel("True Disease")
    axes[0].tick_params(axis="x", rotation=45)

    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Greens",
        xticklabels=class_names, yticklabels=class_names,
        ax=axes[1], cbar=False,
    )
    axes[1].set_title(f"{title} (Normalized)")
    axes[1].set_xlabel("Predicted Disease")
    axes[1].set_ylabel("True Disease")
    axes[1].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {path}")
    return path
