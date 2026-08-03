"""
Medical AI Engine — Overlay Generator

Module 15.1: Explainability Framework Architecture

Blends colormapped heatmaps with original medical images using alpha transparency.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration

logger = logging.getLogger("OverlayGenerator")


class IOverlayGenerator(ABC):
    """Abstract contract for OverlayGenerator."""

    @abstractmethod
    def generate_overlay(
        self,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        alpha: float = 0.5,
    ) -> Image.Image:
        """Blend original image and heatmap image into an overlay PIL Image."""
        pass


class OverlayGenerator(IOverlayGenerator):
    """
    Blends original medical images with heatmaps using alpha transparency.
    """

    def __init__(self, config: Optional[ExplainabilityConfiguration] = None) -> None:
        self.config = config or ExplainabilityConfiguration.default()

    def generate_overlay(
        self,
        original_image: Image.Image,
        heatmap_image: Image.Image,
        alpha: float = 0.5,
    ) -> Image.Image:
        """
        Args:
            original_image: PIL Image of the original scan.
            heatmap_image: PIL Image (RGB) of the colormapped heatmap.
            alpha: Transparency factor (0.0 = original image only, 1.0 = heatmap only).

        Returns:
            Blended PIL Image in RGB format.
        """
        orig_rgb = original_image.convert("RGB").resize(heatmap_image.size)
        heat_rgb = heatmap_image.convert("RGB")

        # Alpha blend: overlay = (1 - alpha) * original + alpha * heatmap
        blended = Image.blend(orig_rgb, heat_rgb, alpha=alpha)
        return blended
