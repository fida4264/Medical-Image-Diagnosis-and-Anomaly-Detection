"""
Medical AI Engine — Overlay Composer

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Orchestrates color mapping, alpha blending, attention mask generation, region extraction,
bounding region detection, and quality metrics check into a unified visualization bundle.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Union
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.visualization.overlay_configuration import OverlayConfiguration
from medical_ai_engine.explainability.visualization.overlay_metadata import OverlayMetadata
from medical_ai_engine.explainability.visualization.color_mapper import ColorMapper
from medical_ai_engine.explainability.visualization.alpha_blender import AlphaBlender
from medical_ai_engine.explainability.visualization.attention_mask_generator import AttentionMaskGenerator
from medical_ai_engine.explainability.visualization.bounding_region_detector import BoundingRegionDetector
from medical_ai_engine.explainability.visualization.region_extractor import RegionExtractor
from medical_ai_engine.explainability.visualization.visualization_quality_checker import VisualizationQualityChecker

logger = logging.getLogger("OverlayComposer")


class OverlayComposer:
    """
    Master pipeline composer assembling all visualization elements.
    """

    def __init__(self, config: OverlayConfiguration | None = None) -> None:
        self.config = config or OverlayConfiguration.default()
        self.color_mapper = ColorMapper()
        self.alpha_blender = AlphaBlender()
        self.mask_generator = AttentionMaskGenerator()
        self.bounding_detector = BoundingRegionDetector()
        self.region_extractor = RegionExtractor()
        self.quality_checker = VisualizationQualityChecker()

    def compose(
        self,
        original_image: Union[Image.Image, np.ndarray],
        heatmap_2d: np.ndarray,
        config_override: OverlayConfiguration | None = None,
    ) -> Dict[str, Any]:
        """
        Execute full visualization composition pipeline.

        Args:
            original_image: Original input image (PIL Image or uint8 RGB numpy array).
            heatmap_2d: Normalized 2D Grad-CAM heatmap array [0.0, 1.0].
            config_override: Optional per-call OverlayConfiguration override.

        Returns:
            Dict bundle containing:
                colored_heatmap_pil: PIL.Image
                colored_heatmap_arr: np.ndarray (H, W, 3)
                overlay_pil: PIL.Image
                overlay_arr: np.ndarray (H, W, 3)
                attention_mask_pil: PIL.Image
                attention_mask_arr: np.ndarray (H, W)
                important_regions: List[dict]
                quality_metrics: Dict[str, float]
                metadata: OverlayMetadata
        """
        t0 = time.perf_counter()
        cfg = config_override or self.config

        # 1. Color Mapping: 2D heatmap -> 3D RGB colored heatmap
        heat_rgb_arr, heat_rgb_pil = self.color_mapper.apply_colormap(
            heatmap_2d=heatmap_2d,
            colormap_name=cfg.colormap,
        )

        # 2. Alpha Blending: Original image + Colored heatmap -> Overlay image
        overlay_arr, overlay_pil = self.alpha_blender.blend(
            original_image=original_image,
            colored_heatmap=heat_rgb_pil,
            alpha=cfg.alpha,
        )

        # 3. Attention Mask Generation
        binary_mask_arr, binary_mask_pil = self.mask_generator.generate_mask(
            heatmap_2d=heatmap_2d,
            threshold=cfg.mask_threshold,
        )

        # 4. Region Extraction & Bounding Region Detection
        important_regions = self.region_extractor.extract_regions(
            heatmap_2d=heatmap_2d,
            binary_mask=binary_mask_arr,
            threshold=cfg.mask_threshold,
        )

        # Optionally draw bounding boxes on overlay image
        if cfg.draw_bounding_boxes and important_regions:
            overlay_pil = self.bounding_detector.draw_bounding_boxes(
                overlay_pil=overlay_pil,
                regions=important_regions,
                box_color=cfg.bounding_box_color,
                thickness=cfg.bounding_box_thickness,
            )
            overlay_arr = np.array(overlay_pil, dtype=np.uint8)

        # 5. Visualization Quality Checker
        orig_arr = np.array(original_image.convert("RGB")) if isinstance(original_image, Image.Image) else original_image
        quality_metrics = self.quality_checker.check_quality(
            original_image=orig_arr,
            heatmap_2d=heatmap_2d,
            overlay_image=overlay_arr,
            threshold=cfg.mask_threshold,
        )

        # 6. Build Metadata
        meta = OverlayMetadata(
            colormap=cfg.colormap,
            alpha=cfg.alpha,
            mask_threshold=cfg.mask_threshold,
            image_width=overlay_pil.width,
            image_height=overlay_pil.height,
            region_count=len(important_regions),
            max_activation_peak=float(heatmap_2d.max()) if heatmap_2d is not None else 0.0,
            quality_metrics=quality_metrics,
        )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
        quality_metrics["composition_time_ms"] = elapsed_ms

        logger.info(
            "[OverlayComposer] Composed overlay (colormap='%s', alpha=%.2f, regions=%d, time=%.1fms)",
            cfg.colormap, cfg.alpha, len(important_regions), elapsed_ms,
        )

        return {
            "colored_heatmap_pil": heat_rgb_pil,
            "colored_heatmap_arr": heat_rgb_arr,
            "overlay_pil": overlay_pil,
            "overlay_arr": overlay_arr,
            "attention_mask_pil": binary_mask_pil,
            "attention_mask_arr": binary_mask_arr,
            "important_regions": important_regions,
            "quality_metrics": quality_metrics,
            "metadata": meta,
        }
