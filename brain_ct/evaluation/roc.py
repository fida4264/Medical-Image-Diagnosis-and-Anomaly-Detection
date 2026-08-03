"""
brain_ct/evaluation/roc.py
===========================
ROC curves for Brain CT disease classification.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

logger = logging.getLogger(__name__)


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_dir: str | Path = "saved_models/brain_ct/roc_curves",
) -> Dict[str, float]:
    """Computes and plots One-vs-Rest ROC curves for each disease class."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "brain_ct_roc_curves.png"

    auc_scores = {}
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, cls_name in enumerate(class_names):
        y_binary = (y_true == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
            roc_auc = float(auc(fpr, tpr))
        else:
            roc_auc = 1.0
            fpr, tpr = np.array([0.0, 1.0]), np.array([1.0, 1.0])
        auc_scores[cls_name] = roc_auc
        ax.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Brain CT Disease Classification — ROC Curves")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curves to {png_path}")
    return auc_scores
