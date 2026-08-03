"""
modality_classifier/utils/visualization.py
===========================================
Visualization utilities for training curves.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def plot_training_curves(history: Dict[str, List[float]], save_path: str | Path = "outputs/modality_training_curves.png") -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "s-", label="Val Loss")
    axes[0].set_title("Loss vs. Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "o-", label="Train Acc")
    axes[1].plot(epochs, history["val_acc"], "s-", label="Val Acc")
    if "val_f1" in history:
        axes[1].plot(epochs, history["val_f1"], "^--", label="Val Macro F1")
    axes[1].set_title("Accuracy & F1 vs. Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved training curves to {path}")
    return path
