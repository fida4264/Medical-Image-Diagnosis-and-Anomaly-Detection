"""
Medical AI Engine — Attention Mask Generator

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Thresholds 2D normalized Grad-CAM heatmaps at a specified activation cutoff
(e.g., threshold = 0.2) to isolate primary attention regions as binary masks.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("AttentionMaskGenerator")


class AttentionMaskGenerator:
    """
    Generates binary attention masks from 2D float heatmaps.
    """

    def generate_mask(
        self,
        heatmap_2d: np.ndarray,
        threshold: float = 0.2,
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Create binary attention mask (255 where heatmap >= threshold, 0 elsewhere).

        Args:
            heatmap_2d: 2D float array (H, W) in range [0, 1].
            threshold: Intensity cutoff threshold in range [0.0, 1.0]. Default: 0.2.

        Returns:
            Tuple of (binary_uint8_array: shape (H, W) with values 0 or 255, pil_image: PIL L-mode Image).
        """
        if heatmap_2d is None or heatmap_2d.ndim != 2:
            h, w = (224, 224) if heatmap_2d is None else heatmap_2d.shape[:2]
            mask_arr = np.zeros((h, w), dtype=np.uint8)
            return mask_arr, Image.fromarray(mask_arr, mode="L")

        thresh = float(np.clip(threshold, 0.0, 1.0))
        binary_mask = (heatmap_2d >= thresh).astype(np.uint8) * 255
        pil_mask = Image.fromarray(binary_mask, mode="L")

        logger.debug(
            "[AttentionMaskGenerator] Generated binary mask with threshold=%.2f (active area: %.1f%%)",
            thresh, (np.sum(binary_mask > 0) / binary_mask.size) * 100.0,
        )
        return binary_mask, pil_mask
