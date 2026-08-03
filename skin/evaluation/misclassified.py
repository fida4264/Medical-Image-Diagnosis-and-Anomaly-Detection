"""
skin/evaluation/misclassified.py
=================================
Misclassified samples grid visualization for Module 12.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


def save_misclassified_samples(
    dataset: Dataset,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str | Path = "outputs/skin/misclassified_samples.png",
    max_samples: int = 16,
) -> None:
    """Save grid plot of misclassified test samples."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    mis_indices = np.where(y_true != y_pred)[0]
    if len(mis_indices) == 0:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "Zero misclassifications! 100% test accuracy.", ha="center", va="center")
        ax.axis("off")
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        return

    n = min(len(mis_indices), max_samples)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(-1)

    for i in range(n):
        idx = mis_indices[i]
        img_tensor, _ = dataset[idx]
        img_np = img_tensor.permute(1, 2, 0).numpy()
        # Un-normalize ImageNet
        img_np = (img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])).clip(0, 1)

        axes[i].imshow(img_np)
        true_cls = class_names[y_true[idx]] if y_true[idx] < len(class_names) else str(y_true[idx])
        pred_cls = class_names[y_pred[idx]] if y_pred[idx] < len(class_names) else str(y_pred[idx])
        axes[i].set_title(f"True: {true_cls}\nPred: {pred_cls}", fontsize=8, color="red")
        axes[i].axis("off")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.suptitle("Module 12: Misclassified Skin Lesion Samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
