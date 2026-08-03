"""
chest_xray/evaluation/misclassified.py
========================================
Misclassified sample extraction and visual grid for Chest X-ray model.
Mirrors Module 7 pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


def extract_misclassified_samples(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    image_paths: List[str],
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Extracts all misclassified sample details."""
    misclassified = []
    for idx in range(len(y_true)):
        t, p = int(y_true[idx]), int(y_pred[idx])
        if t != p:
            misclassified.append({
                "sample_index": idx,
                "image_path": image_paths[idx] if idx < len(image_paths) else "",
                "true_class": class_names[t] if t < len(class_names) else str(t),
                "predicted_class": class_names[p] if p < len(class_names) else str(p),
                "confidence": float(y_probs[idx, p]),
                "true_class_probability": float(y_probs[idx, t]),
            })
    logger.info(f"Found {len(misclassified)} misclassified samples out of {len(y_true)}.")
    return misclassified


def plot_misclassified_grid(
    misclassified_samples: List[Dict[str, Any]],
    save_path: str | Path = "saved_models/chest_xray/evaluation/misclassified/grid.png",
    max_images: int = 12,
) -> Optional[Path]:
    if not misclassified_samples:
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = misclassified_samples[:max_images]
    cols, rows = 4, int(np.ceil(len(samples) / 4))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < len(samples):
            s = samples[i]
            img_p = Path(s["image_path"])
            if img_p.is_file():
                try:
                    ax.imshow(Image.open(img_p).convert("RGB"))
                except Exception:
                    ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            ax.set_title(
                f"True: {s['true_class']}\nPred: {s['predicted_class']}\n{s['confidence']*100:.1f}%",
                color="red", fontsize=9,
            )
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved misclassified grid to {path}")
    return path
