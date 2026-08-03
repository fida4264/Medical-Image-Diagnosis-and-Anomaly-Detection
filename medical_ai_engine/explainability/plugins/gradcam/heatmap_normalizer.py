"""
Medical AI Engine — Grad-CAM Heatmap Normalizer

Module 15.3: Grad-CAM Engine

Resizes raw 2D activation map to requested spatial resolution (e.g. 224x224)
and normalizes intensity values to range [0.0, 1.0]. Handles constant/flat
maps safely without division-by-zero or NaN output.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np

logger = logging.getLogger("HeatmapNormalizer")


class HeatmapNormalizer:
    """
    Resizes and normalizes raw activation maps into standard 2D heatmaps [0.0, 1.0].
    """

    def normalize(
        self,
        raw_cam: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Bilinear resize and min-max normalize a 2D raw activation map.

        Args:
            raw_cam: 2D numpy array of shape (H_in, W_in).
            target_size: Desired spatial resolution (H_out, W_out).
            normalize: Min-max normalize to [0, 1].

        Returns:
            2D numpy float32 array of shape (H_out, W_out) with values in [0.0, 1.0].
        """
        if raw_cam is None:
            return np.zeros(target_size, dtype=np.float32)

        # 1. Bilinear Interpolation / Spatial Resizing
        resized = self._resize_2d(raw_cam, target_size)

        # 2. Min-Max Normalization
        if not normalize:
            return resized.astype(np.float32)

        cam_min = resized.min()
        cam_max = resized.max()
        denom = cam_max - cam_min

        if denom < 1e-8 or np.isnan(denom):
            # Flat/constant activation map
            logger.debug("[HeatmapNormalizer] Constant activation map detected (max-min < 1e-8). Returning zeros.")
            return np.zeros(target_size, dtype=np.float32)

        normalized = (resized - cam_min) / denom
        return np.clip(normalized, 0.0, 1.0).astype(np.float32)

    def _resize_2d(self, arr: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """Bilinear spatial resize of 2D array."""
        h_out, w_out = target_size
        h_in, w_in = arr.shape[:2]

        if (h_in, w_in) == (h_out, w_out):
            return arr.copy()

        # OpenCV fallback if installed
        try:
            import cv2
            return cv2.resize(arr, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
        except ImportError:
            pass

        # PIL fallback
        try:
            from PIL import Image
            # Normalize to uint8 temporarily for PIL resize or convert directly float32
            img = Image.fromarray(arr.astype(np.float32), mode="F")
            resized_img = img.resize((w_out, h_out), resample=Image.BILINEAR)
            return np.array(resized_img, dtype=np.float32)
        except ImportError:
            pass

        # Pure NumPy nearest-neighbor fallback
        row_indices = np.linspace(0, h_in - 1, h_out).astype(int)
        col_indices = np.linspace(0, w_in - 1, w_out).astype(int)
        return arr[np.ix_(row_indices, col_indices)]
