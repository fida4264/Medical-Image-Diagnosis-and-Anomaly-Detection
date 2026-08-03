"""
Medical AI Engine — Visualization Quality Checker

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Computes quantitative quality metrics for generated visualizations:
  - Overlay Coverage Ratio
  - Heatmap Sharpness (Spatial Variance / Gradient Variance)
  - Activation Contrast Ratio
  - Overlay Visibility Contrast
  - Color Distribution & Entropy
"""

from __future__ import annotations

import logging
from typing import Any, Dict
import numpy as np

from medical_ai_engine.explainability.visualization.overlay_exceptions import (
    VisualizationQualityError,
)

logger = logging.getLogger("VisualizationQualityChecker")


class VisualizationQualityChecker:
    """
    Computes quantitative validation metrics for generated heatmaps and overlays.
    """

    def check_quality(
        self,
        original_image: np.ndarray,
        heatmap_2d: np.ndarray,
        overlay_image: np.ndarray,
        threshold: float = 0.2,
    ) -> Dict[str, float]:
        """
        Evaluate visualization quality metrics.

        Args:
            original_image: Original RGB uint8 array (H, W, 3).
            heatmap_2d: Normalized 2D float heatmap array (H, W) in [0, 1].
            overlay_image: Blended RGB uint8 overlay array (H, W, 3).
            threshold: Activation cutoff threshold.

        Returns:
            Dict containing quality metric scores:
                coverage_ratio: float [0, 1]
                heatmap_sharpness: float >= 0
                activation_contrast: float >= 0
                overlay_visibility: float [0, 1]
                color_entropy: float >= 0
        """
        try:
            if heatmap_2d is None:
                raise VisualizationQualityError("heatmap_2d is None.")

            # 1. Overlay Coverage Ratio
            active_pixels = np.sum(heatmap_2d >= threshold)
            total_pixels = heatmap_2d.size
            coverage_ratio = float(active_pixels / total_pixels) if total_pixels > 0 else 0.0

            # 2. Heatmap Sharpness (Spatial Variance of Gradients)
            gy, gx = np.gradient(heatmap_2d)
            grad_magnitude = np.sqrt(gx**2 + gy**2)
            heatmap_sharpness = float(np.std(grad_magnitude))

            # 3. Activation Contrast (Max - Min) / (Mean + eps)
            h_min, h_max = float(heatmap_2d.min()), float(heatmap_2d.max())
            h_mean = float(heatmap_2d.mean())
            activation_contrast = float((h_max - h_min) / (h_mean + 1e-6))

            # 4. Overlay Visibility (RMS contrast difference between original & overlay)
            if original_image is not None and overlay_image is not None:
                orig_gray = np.mean(original_image.astype(np.float32), axis=-1) if original_image.ndim == 3 else original_image.astype(np.float32)
                over_gray = np.mean(overlay_image.astype(np.float32), axis=-1) if overlay_image.ndim == 3 else overlay_image.astype(np.float32)
                diff = np.abs(over_gray - orig_gray)
                overlay_visibility = float(np.mean(diff) / 255.0)
            else:
                overlay_visibility = 0.5

            # 5. Color Distribution & Channel Entropy
            if overlay_image is not None and overlay_image.ndim == 3:
                std_r = np.std(overlay_image[:, :, 0])
                std_g = np.std(overlay_image[:, :, 1])
                std_b = np.std(overlay_image[:, :, 2])
                color_entropy = float((std_r + std_g + std_b) / 3.0)
            else:
                color_entropy = 0.0

            metrics = {
                "coverage_ratio": round(coverage_ratio, 4),
                "heatmap_sharpness": round(heatmap_sharpness, 4),
                "activation_contrast": round(activation_contrast, 4),
                "overlay_visibility": round(overlay_visibility, 4),
                "color_entropy": round(color_entropy, 4),
            }

            logger.debug("[VisualizationQualityChecker] Computed quality metrics: %s", metrics)
            return metrics
        except VisualizationQualityError:
            raise
        except Exception as exc:
            raise VisualizationQualityError(f"Unexpected error calculating quality metrics: {exc}") from exc
