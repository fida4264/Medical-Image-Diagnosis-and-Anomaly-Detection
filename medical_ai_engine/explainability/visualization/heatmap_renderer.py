"""
Medical AI Engine — Heatmap Renderer Facade

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Primary entry point for the visualization subsystem.
Transforms normalized Grad-CAM heatmaps into clinically readable overlay images,
attention masks, geometric bounding regions, and quality metrics without generating
any clinical or medical text explanations (text generation belongs to Module 15.5).
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Union
import numpy as np
from PIL import Image

from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.visualization.overlay_configuration import OverlayConfiguration
from medical_ai_engine.explainability.visualization.overlay_composer import OverlayComposer
from medical_ai_engine.explainability.visualization.image_exporter import ImageExporter

logger = logging.getLogger("HeatmapRenderer")


class HeatmapRenderer:
    """
    Primary facade for heatmap rendering, overlay blending, region extraction, and image export.
    """

    def __init__(self, config: Optional[OverlayConfiguration] = None) -> None:
        self.config = config or OverlayConfiguration.default()
        self.composer = OverlayComposer(config=self.config)
        self.exporter = ImageExporter()

    def render_visualization(
        self,
        image: Union[Image.Image, np.ndarray],
        normalized_heatmap: np.ndarray,
        context: Optional[ExplainabilityContext] = None,
        config_override: Optional[OverlayConfiguration] = None,
        export_dir: str = "",
    ) -> Dict[str, Any]:
        """
        Render visualization bundle from input image and normalized 2D Grad-CAM heatmap.

        Args:
            image: Original input image (PIL Image or uint8 RGB numpy array).
            normalized_heatmap: 2D numpy array [0.0, 1.0].
            context: Shared ExplainabilityContext to populate.
            config_override: Optional OverlayConfiguration override.
            export_dir: Optional directory to save output PNG/JPEG files.

        Returns:
            Dict containing:
                overlay_image: PIL.Image
                heatmap_image: PIL.Image
                attention_mask_image: PIL.Image
                important_regions: List[dict]
                visualization_metrics: Dict[str, float]
                overlay_metadata: dict
                saved_paths: Dict[str, str]
        """
        t0 = time.perf_counter()
        cfg = config_override or self.config

        # 1. Compose full visualization bundle
        bundle = self.composer.compose(
            original_image=image,
            heatmap_2d=normalized_heatmap,
            config_override=cfg,
        )

        # 2. Export image artifacts if export_dir is set or requested via config
        target_dir = export_dir or cfg.export_dir
        saved_paths: Dict[str, str] = {}
        if target_dir:
            req_id = context.request_id if context else "xai"
            saved_paths = self.exporter.export(
                composition_bundle=bundle,
                export_dir=target_dir,
                filename_prefix=req_id,
                export_format=cfg.export_format,
                context=context,
            )

        # 3. Populate ExplainabilityContext
        if context is not None:
            context.overlay_image = bundle["overlay_pil"]
            context.important_regions = bundle["important_regions"]
            context.future_extensions["visualization_metrics"] = bundle["quality_metrics"]
            context.future_extensions["overlay_metadata"] = bundle["metadata"].to_dict()
            if saved_paths:
                context.future_extensions["saved_visualization_paths"] = saved_paths

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 3)
            context.record_metric("visualization_render_ms", elapsed_ms)
            context.log(
                f"Visualization rendered (colormap='{cfg.colormap}', alpha={cfg.alpha}, regions={len(bundle['important_regions'])})",
                stage="HeatmapRenderer",
            )

        return {
            "overlay_image": bundle["overlay_pil"],
            "heatmap_image": bundle["colored_heatmap_pil"],
            "attention_mask_image": bundle["attention_mask_pil"],
            "important_regions": bundle["important_regions"],
            "visualization_metrics": bundle["quality_metrics"],
            "overlay_metadata": bundle["metadata"].to_dict(),
            "saved_paths": saved_paths,
        }
