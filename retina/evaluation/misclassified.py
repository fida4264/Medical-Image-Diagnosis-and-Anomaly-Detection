"""
retina/evaluation/misclassified.py
===================================
Misclassified samples grid saver for retina evaluation.
Mirrors Module 7/8/9/10 misclassified.py pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def save_misclassified_grid(
    all_images: List,
    all_labels: List[int],
    all_preds: List[int],
    inv_label_map: dict,
    output_path: Path,
    max_samples: int = 16,
) -> None:
    """Save a grid of misclassified retina images to disk."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        misclassified = [
            (img, true, pred)
            for img, true, pred in zip(all_images, all_labels, all_preds)
            if true != pred
        ]

        if not misclassified:
            logger.info("No misclassified samples found — skipping grid.")
            return

        samples = misclassified[:max_samples]
        n = len(samples)
        cols = min(4, n)
        rows = (n + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
        axes = np.array(axes).reshape(-1) if n > 1 else [axes]

        for ax, (img, true_lbl, pred_lbl) in zip(axes, samples):
            img_arr = np.array(img)
            if img_arr.ndim == 3 and img_arr.shape[0] == 3:
                img_arr = np.transpose(img_arr, (1, 2, 0))
            img_arr = (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8)
            ax.imshow(img_arr)
            ax.set_title(
                f"T:{inv_label_map.get(true_lbl,'?')}\nP:{inv_label_map.get(pred_lbl,'?')}",
                fontsize=7, color="red"
            )
            ax.axis("off")

        for ax in axes[len(samples):]:
            ax.axis("off")

        plt.suptitle("Retina — Misclassified Samples", fontsize=12)
        plt.tight_layout()
        plt.savefig(output_path, dpi=120)
        plt.close(fig)
        logger.info(f"Misclassified grid saved to '{output_path}'.")

    except Exception as e:
        logger.warning(f"Misclassified grid generation skipped: {e}")
