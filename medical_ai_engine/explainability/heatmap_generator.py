"""
Medical AI Engine — Heatmap Generator

Module 15.1: Explainability Framework Architecture

Converts raw 2D activation tensors into normalized, colormapped heatmaps.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration

logger = logging.getLogger("HeatmapGenerator")


class IHeatmapGenerator(ABC):
    """Abstract contract for HeatmapGenerator."""

    @abstractmethod
    def generate_heatmap(
        self,
        activation_map: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        colormap_name: str = "JET",
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Convert activation map to colormapped heatmap array and PIL Image.
        """
        pass


class HeatmapGenerator(IHeatmapGenerator):
    """
    Standard implementation converting activation maps into RGB heatmaps.
    """

    def __init__(self, config: Optional[ExplainabilityConfiguration] = None) -> None:
        self.config = config or ExplainabilityConfiguration.default()

    def generate_heatmap(
        self,
        activation_map: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        colormap_name: str = "JET",
    ) -> Tuple[np.ndarray, Image.Image]:
        """
        Args:
            activation_map: 2D array of shape (H, W) in [0.0, 1.0].
            target_size: (W, H) tuple.
            colormap_name: OpenCV / Matplotlib colormap name.

        Returns:
            Tuple of (RGB uint8 numpy array, PIL Image).
        """
        # Ensure 2D
        if activation_map.ndim > 2:
            activation_map = np.squeeze(activation_map)

        # Normalize 0..1
        min_val, max_val = activation_map.min(), activation_map.max()
        if max_val > min_val:
            norm_map = (activation_map - min_val) / (max_val - min_val)
        else:
            norm_map = np.zeros_like(activation_map)

        # Resize PIL Image
        pil_grayscale = Image.fromarray((norm_map * 255).astype(np.uint8), mode="L")
        pil_resized = pil_grayscale.resize(target_size, resample=Image.Resampling.BILINEAR)

        # Convert to pseudo-RGB heatmap
        resized_arr = np.array(pil_resized) / 255.0

        # Simple colormap mapping (Blue -> Green -> Red)
        r = np.clip(2.0 * resized_arr - 0.5, 0.0, 1.0)
        g = np.clip(1.0 - 2.0 * np.abs(resized_arr - 0.5), 0.0, 1.0)
        b = np.clip(1.0 - 2.0 * resized_arr, 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        rgb_uint8 = (rgb * 255).astype(np.uint8)

        pil_rgb = Image.fromarray(rgb_uint8, mode="RGB")
        return rgb_uint8, pil_rgb
