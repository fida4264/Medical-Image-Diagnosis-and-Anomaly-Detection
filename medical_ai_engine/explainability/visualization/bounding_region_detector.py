"""
Medical AI Engine — Bounding Region Detector

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Detects geometric bounding boxes, contour polygons, and mask region bounds for
high-activation areas without performing any clinical or disease interpretation.
"""

from __future__ import annotations

import logging
from typing import Any, List, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw

logger = logging.getLogger("BoundingRegionDetector")


class BoundingRegionDetector:
    """
    Detects geometric bounding boxes and contour polygons for activation regions.
    """

    def detect_bounding_regions(
        self,
        binary_mask: np.ndarray,
        heatmap_2d: np.ndarray,
        threshold: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Detect connected activation components and compute bounding boxes and polygons.

        Args:
            binary_mask: 2D uint8 binary mask array (0 or 255).
            heatmap_2d: 2D float heatmap array [0, 1].
            threshold: Intensity threshold.

        Returns:
            List of region dicts:
                [{
                    "region_id": int,
                    "bbox": [x_min, y_min, x_max, y_max],
                    "centroid": [x_center, y_center],
                    "area_pixels": int,
                    "area_ratio": float,
                    "peak_intensity": float,
                    "mean_intensity": float,
                    "polygon": list of [x, y] coordinates
                }]
        """
        if binary_mask is None or np.sum(binary_mask) == 0:
            return []

        # Try OpenCV contour detection
        try:
            import cv2
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for idx, cnt in enumerate(contours):
                area = cv2.contourArea(cnt)
                if area < 5:  # Ignore tiny noise spots
                    continue

                x, y, w, h = cv2.boundingRect(cnt)
                bbox = [int(x), int(y), int(x + w), int(y + h)]

                # Peak and mean inside box
                sub_map = heatmap_2d[y:y+h, x:x+w]
                peak_val = float(np.max(sub_map)) if sub_map.size > 0 else 0.0
                mean_val = float(np.mean(sub_map)) if sub_map.size > 0 else 0.0

                # Centroid
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = int(x + w / 2), int(y + h / 2)

                # Polygon approximation
                approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
                polygon = [[int(pt[0][0]), int(pt[0][1])] for pt in approx]

                regions.append({
                    "region_id": idx + 1,
                    "bbox": bbox,
                    "centroid": [cx, cy],
                    "area_pixels": int(area),
                    "area_ratio": float(area / (binary_mask.shape[0] * binary_mask.shape[1])),
                    "peak_intensity": round(peak_val, 4),
                    "mean_intensity": round(mean_val, 4),
                    "polygon": polygon,
                })

            # Sort regions descending by area
            regions.sort(key=lambda r: r["area_pixels"], reverse=True)
            return regions
        except ImportError:
            pass

        # NumPy bounding box fallback
        return self._numpy_region_fallback(binary_mask, heatmap_2d)

    def draw_bounding_boxes(
        self,
        overlay_pil: Image.Image,
        regions: List[Dict[str, Any]],
        box_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2,
    ) -> Image.Image:
        """Draw geometric bounding box rectangles on overlay PIL Image."""
        if not regions or overlay_pil is None:
            return overlay_pil

        draw_img = overlay_pil.copy()
        draw = ImageDraw.Draw(draw_img)

        for region in regions:
            bbox = region.get("bbox", None)
            if bbox and len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                for i in range(thickness):
                    draw.rectangle(
                        [x_min - i, y_min - i, x_max + i, y_max + i],
                        outline=box_color,
                    )
        return draw_img

    def _numpy_region_fallback(self, binary_mask: np.ndarray, heatmap_2d: np.ndarray) -> List[Dict[str, Any]]:
        """NumPy fallback bounding box for binary mask."""
        active_coords = np.argwhere(binary_mask > 0)
        if len(active_coords) == 0:
            return []

        y_min, x_min = active_coords.min(axis=0)
        y_max, x_max = active_coords.max(axis=0)

        cx, cy = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
        area = int(len(active_coords))
        sub_map = heatmap_2d[y_min:y_max+1, x_min:x_max+1]

        bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        polygon = [[int(x_min), int(y_min)], [int(x_max), int(y_min)], [int(x_max), int(y_max)], [int(x_min), int(y_max)]]

        return [{
            "region_id": 1,
            "bbox": bbox,
            "centroid": [cx, cy],
            "area_pixels": area,
            "area_ratio": float(area / binary_mask.size),
            "peak_intensity": float(np.max(sub_map)) if sub_map.size > 0 else 0.0,
            "mean_intensity": float(np.mean(sub_map)) if sub_map.size > 0 else 0.0,
            "polygon": polygon,
        }]
