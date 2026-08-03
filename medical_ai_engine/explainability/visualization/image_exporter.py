"""
Medical AI Engine — Image Exporter

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Exports visualization image artifacts (overlay.png, heatmap.png, attention_mask.png)
and metadata JSON files to disk. Updates file paths in ExplainabilityContext.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional
from PIL import Image

from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
from medical_ai_engine.explainability.visualization.overlay_exceptions import ExportError
from medical_ai_engine.explainability.visualization.overlay_metadata import OverlayMetadata

logger = logging.getLogger("ImageExporter")


class ImageExporter:
    """
    Exports visualization image artifacts and metadata to disk.
    """

    def export(
        self,
        composition_bundle: Dict[str, Any],
        export_dir: str,
        filename_prefix: str = "gradcam",
        export_format: str = "PNG",
        context: Optional[ExplainabilityContext] = None,
    ) -> Dict[str, str]:
        """
        Save overlay.png, heatmap.png, attention_mask.png, and overlay_metadata.json.

        Args:
            composition_bundle: Output dictionary from OverlayComposer.
            export_dir: Target output directory path.
            filename_prefix: Base prefix for saved files.
            export_format: 'PNG' or 'JPEG'.
            context: Shared ExplainabilityContext to update with saved file paths.

        Returns:
            Dict mapping artifact key -> file path string.
        """
        if not export_dir:
            logger.debug("[ImageExporter] No export_dir provided. Skipping disk write.")
            return {}

        try:
            os.makedirs(export_dir, exist_ok=True)
            ext = "jpg" if export_format.upper() in ("JPEG", "JPG") else "png"
            fmt = "JPEG" if ext == "jpg" else "PNG"

            paths: Dict[str, str] = {}

            # 1. Export Overlay Image
            overlay_pil: Optional[Image.Image] = composition_bundle.get("overlay_pil")
            if overlay_pil is not None:
                overlay_path = os.path.join(export_dir, f"{filename_prefix}_overlay.{ext}")
                overlay_pil.save(overlay_path, format=fmt)
                paths["overlay_path"] = overlay_path

            # 2. Export Heatmap Image
            heatmap_pil: Optional[Image.Image] = composition_bundle.get("colored_heatmap_pil")
            if heatmap_pil is not None:
                heatmap_path = os.path.join(export_dir, f"{filename_prefix}_heatmap.{ext}")
                heatmap_pil.save(heatmap_path, format=fmt)
                paths["heatmap_path"] = heatmap_path

            # 3. Export Attention Mask Image
            mask_pil: Optional[Image.Image] = composition_bundle.get("attention_mask_pil")
            if mask_pil is not None:
                mask_path = os.path.join(export_dir, f"{filename_prefix}_mask.{ext}")
                mask_pil.save(mask_path, format=fmt)
                paths["attention_mask_path"] = mask_path

            # 4. Export Overlay Metadata JSON
            metadata_obj: Optional[OverlayMetadata] = composition_bundle.get("metadata")
            if metadata_obj is not None:
                meta_dict = metadata_obj.to_dict()
                meta_dict["saved_paths"] = paths
                meta_dict["important_regions"] = composition_bundle.get("important_regions", [])
                meta_path = os.path.join(export_dir, f"{filename_prefix}_metadata.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta_dict, f, indent=2)
                paths["metadata_path"] = meta_path

            # 5. Update ExplainabilityContext if provided
            if context is not None:
                if "overlay_path" in paths:
                    context.overlay_path = paths["overlay_path"]
                if "heatmap_path" in paths:
                    context.heatmap_path = paths["heatmap_path"]
                if "attention_mask_path" in paths:
                    context.future_extensions["attention_mask_path"] = paths["attention_mask_path"]

            logger.info("[ImageExporter] Exported %d visualization files to '%s'.", len(paths), export_dir)
            return paths

        except Exception as exc:
            raise ExportError(filepath=export_dir, reason=str(exc)) from exc
