"""
Medical AI Engine — Alpha Blender

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Blends the original medical image with the RGB colored heatmap using a configurable
transparency ratio alpha in [0.1, 0.9] (default: 0.5).
"""

from __future__ import annotations

import logging
from typing import Tuple, Union
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.visualization.overlay_exceptions import (
    OverlayBlendingError,
)

logger = logging.getLogger("AlphaBlender")


class AlphaBlender:
    """
    Alpha transparency blender combining original medical images with heatmap overlays.
    """

    def blend(
        self,
        original_image: Union[Image.Image, np.ndarray],
        colored_heatmap: Union[Image.Image, np.ndarray],
        alpha: float = 0.5,
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Blend original image and colored heatmap.

        Args:
            original_image: PIL Image or (H, W, 3) uint8 numpy array.
            colored_heatmap: PIL Image or (H, W, 3) uint8 numpy array.
            alpha: Transparency weight for heatmap in range [0.1, 0.9]. Default: 0.5.

        Returns:
            Tuple of (blended_uint8_array: np.ndarray shape (H, W, 3), blended_pil: PIL.Image).
        """
        # Validate alpha ratio
        alpha_clamped = float(np.clip(alpha, 0.1, 0.9))

        # Convert original_image to (H, W, 3) uint8 numpy array
        orig_arr = self._to_rgb_array(original_image)
        heat_arr = self._to_rgb_array(colored_heatmap)

        # Match dimensions if needed
        if orig_arr.shape[:2] != heat_arr.shape[:2]:
            h, w = orig_arr.shape[:2]
            heat_img = Image.fromarray(heat_arr).resize((w, h), resample=Image.BILINEAR)
            heat_arr = np.array(heat_img, dtype=np.uint8)

        # Alpha blending calculation: alpha * heatmap + (1 - alpha) * original
        blended_float = alpha_clamped * heat_arr.astype(np.float32) + (1.0 - alpha_clamped) * orig_arr.astype(np.float32)
        blended_uint8 = np.clip(blended_float, 0, 255).astype(np.uint8)

        blended_pil = Image.fromarray(blended_uint8, mode="RGB")
        logger.debug("[AlphaBlender] Successfully blended overlay with alpha=%.2f", alpha_clamped)
        return blended_uint8, blended_pil

    def _to_rgb_array(self, img_input: Union[Image.Image, np.ndarray]) -> np.ndarray:
        """Convert input image to 3D RGB uint8 array."""
        if isinstance(img_input, Image.Image):
            rgb_img = img_input.convert("RGB")
            return np.array(rgb_img, dtype=np.uint8)

        if isinstance(img_input, np.ndarray):
            arr = img_input
            if arr.ndim == 2:  # Grayscale (H, W) -> stack to (H, W, 3)
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 1:
                arr = np.concatenate([arr] * 3, axis=-1)
            elif arr.ndim == 3 and arr.shape[2] == 4:  # RGBA -> RGB
                arr = arr[:, :, :3]

            if arr.dtype != np.uint8:
                if arr.max() <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = arr.astype(np.uint8)

            return arr

        raise OverlayBlendingError(f"Unsupported image input type: {type(img_input)}")
