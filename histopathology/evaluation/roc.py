"""
histopathology/evaluation/roc.py
===================================
Multi-class One-vs-Rest ROC curve plot for histopathology evaluation.
Mirrors Module 7/8/9 roc.py pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)


def plot_roc_curves(
    all_labels: List[int],
    all_probs: List[List[float]],
    class_names: List[str],
    output_path: Path,
) -> None:
    """Plot One-vs-Rest ROC curves for each class and save to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        num_classes = len(class_names)
        labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))
        probs_arr = np.array(all_probs)

        fig, ax = plt.subplots(figsize=(9, 7))
        colors = plt.cm.tab10(np.linspace(0, 1, num_classes))

        for i, (cls_name, color) in enumerate(zip(class_names, colors)):
            if labels_bin.shape[1] > i:
                fpr, tpr, _ = roc_curve(labels_bin[:, i], probs_arr[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=1.8, label=f"{cls_name} (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title("Histopathology — Multi-Class ROC Curves (OvR)", fontsize=13)
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"ROC curves saved to '{output_path}'.")

    except Exception as e:
        logger.warning(f"ROC plot generation skipped: {e}")
