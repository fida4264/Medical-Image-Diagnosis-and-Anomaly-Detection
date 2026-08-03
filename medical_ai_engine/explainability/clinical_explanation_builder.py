"""
Medical AI Engine — Clinical Explanation Builder

Module 15.1: Explainability Framework Architecture

Analyzes 2D activation maps to extract high-activation anatomical regions
and construct structured clinical textual descriptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
import numpy as np

from medical_ai_engine.explainability.explainability_configuration import ExplainabilityConfiguration

logger = logging.getLogger("ClinicalExplanationBuilder")


class ClinicalExplanationBuilder:
    """
    Extracts anatomical region localization (quadrants/sectors) and builds textual summaries.
    """

    def __init__(self, config: Optional[ExplainabilityConfiguration] = None) -> None:
        self.config = config or ExplainabilityConfiguration.default()

    def build_explanation(
        self,
        activation_map: np.ndarray,
        modality: str,
        predicted_disease: str,
        confidence: float,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Analyze activation map intensity distribution and produce structured regions & summary.

        Args:
            activation_map: Normalized 2D array of shape (H, W).
            modality: Imaging modality string.
            predicted_disease: Diagnosed condition string.
            confidence: Diagnosis confidence score.

        Returns:
            Tuple of (important_regions_list, clinical_summary_string).
        """
        if activation_map is None or activation_map.ndim < 2:
            return [], f"No activation map available for {predicted_disease} ({modality})."

        h, w = activation_map.shape[:2]
        mid_h, mid_w = h // 2, w // 2

        # Compute average intensity per quadrant
        top_left = float(np.mean(activation_map[:mid_h, :mid_w]))
        top_right = float(np.mean(activation_map[:mid_h, mid_w:]))
        bottom_left = float(np.mean(activation_map[mid_h:, :mid_w]))
        bottom_right = float(np.mean(activation_map[mid_h:, mid_w:]))

        quadrants = [
            ("Top-Left Quadrant", top_left),
            ("Top-Right Quadrant", top_right),
            ("Bottom-Left Quadrant", bottom_left),
            ("Bottom-Right Quadrant", bottom_right),
        ]
        quadrants.sort(key=lambda x: x[1], reverse=True)

        important_regions = []
        for rank, (region_name, mean_val) in enumerate(quadrants[:self.config.max_important_regions], 1):
            important_regions.append({
                "rank": rank,
                "region_name": region_name,
                "activation_score": round(mean_val, 4),
                "relative_importance": "High" if rank == 1 else ("Medium" if rank == 2 else "Low"),
            })

        primary_region = quadrants[0][0]
        summary = (
            f"The model's primary attention for diagnosing {predicted_disease} "
            f"({modality.replace('_', ' ').title()}, confidence {confidence:.1%}) "
            f"is concentrated in the {primary_region} (activation score: {quadrants[0][1]:.3f})."
        )

        return important_regions, summary
