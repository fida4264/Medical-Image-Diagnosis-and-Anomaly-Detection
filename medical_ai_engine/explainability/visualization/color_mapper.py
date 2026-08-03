"""
Medical AI Engine — Color Mapper

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Converts a 2D normalized float heatmap [0.0, 1.0] into a 3D RGB uint8 image array
using configurable colormaps (inferno, jet, turbo, viridis, magma).
Default: inferno.
"""

from __future__ import annotations

import logging
from typing import Tuple
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.visualization.overlay_exceptions import (
    ColorMappingError,
)

logger = logging.getLogger("ColorMapper")


class ColorMapper:
    """
    Applies scientific colormaps to 2D activation arrays.
    """

    def apply_colormap(
        self,
        heatmap_2d: np.ndarray,
        colormap_name: str = "inferno",
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Apply colormap to a normalized 2D numpy array [0, 1].

        Args:
            heatmap_2d: 2D float array (H, W) in range [0, 1].
            colormap_name: 'inferno' (default), 'jet', 'turbo', 'viridis', 'magma'.

        Returns:
            Tuple of (rgb_uint8_array: np.ndarray shape (H, W, 3), pil_image: PIL.Image).
        """
        cmap_key = colormap_name.lower()

        # Validate input
        if heatmap_2d is None or heatmap_2d.ndim != 2:
            raise ColorMappingError(colormap_name, "Input heatmap must be a 2D numpy array.")

        # Clip input safely to [0, 1]
        clipped = np.clip(heatmap_2d, 0.0, 1.0).astype(np.float32)

        # 1. Matplotlib Colormap (Primary preferred implementation)
        try:
            import matplotlib.cm as cm
            cmap_obj = getattr(cm, cmap_key, None)
            if cmap_obj is not None:
                rgba = cmap_obj(clipped)  # Shape (H, W, 4), values in [0, 1]
                rgb_uint8 = (rgba[:, :, :3] * 255).astype(np.uint8)
                pil_img = Image.fromarray(rgb_uint8, mode="RGB")
                return rgb_uint8, pil_img
        except ImportError:
            pass

        # 2. OpenCV Colormap Fallback
        try:
            import cv2
            cv2_cmaps = {
                "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
                "jet": cv2.COLORMAP_JET,
                "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
                "viridis": getattr(cv2, "COLORMAP_VIRIDIS", cv2.COLORMAP_JET),
                "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
            }
            code = cv2_cmaps.get(cmap_key, cv2.COLORMAP_JET)
            gray_u8 = (clipped * 255).astype(np.uint8)
            bgr = cv2.applyColorMap(gray_u8, code)
            rgb_uint8 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_uint8, mode="RGB")
            return rgb_uint8, pil_img
        except ImportError:
            pass

        # 3. Pure NumPy Synthetic LUT Fallback (Zero dependencies)
        rgb_uint8 = self._synthetic_lut_colormap(clipped, cmap_key)
        pil_img = Image.fromarray(rgb_uint8, mode="RGB")
        return rgb_uint8, pil_img

    def _synthetic_lut_colormap(self, norm_map: np.ndarray, cmap_name: str) -> np.ndarray:
        """Synthetic RGB Look-Up Table generator for zero-dependency execution."""
        h, w = norm_map.shape
        rgb = np.zeros((h, w, 3), dtype=np.float32)

        x = norm_map
        if cmap_name == "jet":
            # Jet approximation
            rgb[:, :, 0] = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0, 1)  # R
            rgb[:, :, 1] = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0, 1)  # G
            rgb[:, :, 2] = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0, 1)  # B
        elif cmap_name in ("inferno", "magma"):
            # Inferno approximation (black -> purple -> red -> yellow -> white)
            rgb[:, :, 0] = np.clip(x * 1.5, 0, 1)            # R
            rgb[:, :, 1] = np.clip(x**2 * 1.2, 0, 1)          # G
            rgb[:, :, 2] = np.clip(np.sin(x * np.pi), 0, 1)   # B
        elif cmap_name == "viridis":
            # Viridis approximation (purple -> blue -> green -> yellow)
            rgb[:, :, 0] = np.clip(x**2, 0, 1)                # R
            rgb[:, :, 1] = np.clip(np.sin(x * np.pi), 0, 1)   # G
            rgb[:, :, 2] = np.clip(1.0 - x, 0, 1)             # B
        else:  # turbo or fallback
            rgb[:, :, 0] = np.clip(x * 1.2, 0, 1)
            rgb[:, :, 1] = np.clip(np.sin(x * np.pi), 0, 1)
            rgb[:, :, 2] = np.clip(1.0 - x * 0.8, 0, 1)

        return (rgb * 255.0).astype(np.uint8)
