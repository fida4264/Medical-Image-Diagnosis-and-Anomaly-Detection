"""
retina/evaluation/confusion_matrix.py
======================================
Confusion matrix plot helper for retina evaluation.
Mirrors Module 8/9/10 confusion_matrix.py pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def plot_confusion_matrix(
    all_labels: List[int],
    all_preds: List[int],
    class_names: List[str],
    output_path: Path,
    normalize: bool = True,
) -> None:
    """Plot and save a confusion matrix."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from sklearn.metrics import confusion_matrix

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(all_labels, all_preds)
        if normalize:
            cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

        n = len(class_names)
        fig, ax = plt.subplots(figsize=(max(7, n + 2), max(6, n + 1)))
        im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
        plt.colorbar(im, ax=ax)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(class_names, rotation=40, ha="right", fontsize=9)
        ax.set_yticklabels(class_names, fontsize=9)
        ax.set_title("Retina — Confusion Matrix", fontsize=13)
        ax.set_ylabel("True Label", fontsize=11)
        ax.set_xlabel("Predicted Label", fontsize=11)

        thresh = cm.max() / 2.0
        for i in range(n):
            for j in range(n):
                val = f"{cm[i,j]:.2f}" if normalize else str(int(cm[i, j]))
                ax.text(j, i, val, ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black", fontsize=8)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close(fig)
        logger.info(f"Confusion matrix saved to '{output_path}'.")

    except Exception as e:
        logger.warning(f"Confusion matrix generation skipped: {e}")
