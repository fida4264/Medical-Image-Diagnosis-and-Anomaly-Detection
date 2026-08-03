"""
Medical AI Engine — Region Extractor

Module 15.4: Heatmap Visualization & Clinical Overlay Pipeline

Extracts highest activation regions, peak activations, centroids, connected areas,
and intensity statistics for explainability metadata populating.
No disease interpretation or clinical text is generated here.
"""

from __future__ import annotations

import logging
from typing import Any, List, Dict
import numpy as np

from medical_ai_engine.explainability.visualization.bounding_region_detector import (
    BoundingRegionDetector,
)

logger = logging.getLogger("RegionExtractor")


class RegionExtractor:
    """
    Extracts spatial and intensity region descriptors from Grad-CAM heatmaps.
    """

    def __init__(self) -> None:
        self.bounding_detector = BoundingRegionDetector()

    def extract_regions(
        self,
        heatmap_2d: np.ndarray,
        binary_mask: np.ndarray,
        threshold: float = 0.2,
        max_regions: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Extract ranked region descriptors from heatmap and binary mask.

        Args:
            heatmap_2d: 2D float activation map [0, 1].
            binary_mask: 2D binary uint8 mask (0 or 255).
            threshold: Intensity threshold used.
            max_regions: Max number of top regions to return.

        Returns:
            List of region dicts sorted by peak intensity & area.
        """
        if heatmap_2d is None:
            return []

        raw_regions = self.bounding_detector.detect_bounding_regions(
            binary_mask=binary_mask,
            heatmap_2d=heatmap_2d,
            threshold=threshold,
        )

        extracted = []
        for idx, reg in enumerate(raw_regions[:max_regions]):
            extracted.append({
                "rank": idx + 1,
                "region_id": reg.get("region_id", idx + 1),
                "peak_activation": reg.get("peak_intensity", 0.0),
                "mean_activation": reg.get("mean_intensity", 0.0),
                "centroid": reg.get("centroid", [0, 0]),
                "bbox": reg.get("bbox", [0, 0, 0, 0]),
                "area_pixels": reg.get("area_pixels", 0),
                "area_ratio": reg.get("area_ratio", 0.0),
                "polygon": reg.get("polygon", []),
            })

        logger.debug("[RegionExtractor] Extracted %d regions (threshold=%.2f)", len(extracted), threshold)
        return extracted
