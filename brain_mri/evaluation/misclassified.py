"""
brain_mri/evaluation/misclassified.py
======================================
Identifies misclassified samples and generates visual summary grids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Any

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
    """
    Extracts all misclassified sample details.
    """
    misclassified = []
    for idx in range(len(y_true)):
        t_label = int(y_true[idx])
        p_label = int(y_pred[idx])
        if t_label != p_label:
            misclassified.append({
                "sample_index": idx,
                "image_path": image_paths[idx] if idx < len(image_paths) else "",
                "true_class": class_names[t_label] if t_label < len(class_names) else str(t_label),
                "predicted_class": class_names[p_label] if p_label < len(class_names) else str(p_label),
                "confidence": float(y_probs[idx, p_label]),
                "true_class_probability": float(y_probs[idx, t_label]),
            })

    logger.info(f"Identified {len(misclassified)} misclassified samples out of {len(y_true)} test images.")
    return misclassified


def plot_misclassified_grid(
    misclassified_samples: List[Dict[str, Any]],
    save_path: str | Path = "saved_models/brain_mri/evaluation/misclassified/grid.png",
    max_images: int = 12,
) -> Optional[Path]:
    """
    Plots a sample grid of misclassified images.
    """
    if not misclassified_samples:
        logger.info("No misclassified samples to plot.")
        return None

    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    samples = misclassified_samples[:max_images]
    n_samples = len(samples)
    cols = 4
    rows = int(np.ceil(n_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for i in range(rows * cols):
        ax = axes_flat[i]
        if i < n_samples:
            s = samples[i]
            img_p = Path(s["image_path"])
            if img_p.is_file():
                try:
                    img = Image.open(img_p).convert("RGB")
                    ax.imshow(img)
                except Exception:
                    ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))
            else:
                ax.imshow(np.zeros((224, 224, 3), dtype=np.uint8))

            title_str = (
                f"True: {s['true_class']}\n"
                f"Pred: {s['predicted_class']}\n"
                f"Conf: {s['confidence']*100:.1f}%"
            )
            ax.set_title(title_str, color="red", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved misclassified sample grid to {path}")
    return path
