"""
brain_mri/evaluation/roc.py
============================
Multi-class ROC curves and Precision-Recall curves generation for Brain MRI Model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, precision_recall_curve, roc_curve

logger = logging.getLogger(__name__)


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: List[str],
    save_dir: str | Path = "saved_models/brain_mri/roc_curves",
) -> Dict[str, float]:
    """
    Computes and plots One-vs-Rest ROC curves for each disease class.
    Saves plot to PNG and returns AUC scores dict.
    """
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / "brain_mri_roc_curves.png"

    auc_scores = {}
    num_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(8, 7))

    for i in range(num_classes):
        y_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_binary, y_probs[:, i])
        roc_auc = float(auc(fpr, tpr))
        cls_name = class_names[i]
        auc_scores[cls_name] = roc_auc

        ax.plot(fpr, tpr, lw=2, label=f"{cls_name} (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_title("Brain MRI Disease Classification — ROC Curves (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved ROC curves plot to {png_path}")

    return auc_scores
