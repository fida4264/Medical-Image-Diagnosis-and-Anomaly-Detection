"""
quality_evaluator.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Evaluates quantitative heatmap & overlay quality metrics.
"""

from __future__ import annotations

import logging
import numpy as np

from medical_ai_engine.explainability.visualization.visualization_quality_checker import (
    VisualizationQualityChecker,
)
from .verification_metrics import CategoryScore

logger = logging.getLogger(__name__)


class QualityEvaluator:
    """
    Evaluates quantitative visual quality metrics for generated activation maps.

    Computes:
      - Heatmap sharpness (spatial gradient variance)
      - Activation coverage ratio (active pixels / total pixels)
      - Activation contrast ratio
      - Color entropy / channel variance
    """

    def evaluate(self, score: CategoryScore) -> None:
        checker = VisualizationQualityChecker()

        # Synthetic heatmap test 1: Focused Gaussian-like spot (high quality)
        y, x = np.ogrid[:224, :224]
        high_qual_map = np.exp(-((x - 112) ** 2 + (y - 112) ** 2) / (2 * 20 ** 2))

        # Synthetic heatmap test 2: Uniform noise / flat (low quality)
        flat_map = np.full((224, 224), 0.5, dtype=np.float32)

        try:
            orig = np.zeros((224, 224, 3), dtype=np.uint8)
            over = np.repeat((high_qual_map[:, :, None] * 255).astype(np.uint8), 3, axis=2)
            m_high = checker.check_quality(orig, high_qual_map, over)
            if m_high["coverage_ratio"] > 0.01 and m_high["heatmap_sharpness"] > 0.0:
                score.record_pass(f"Focused spot quality check passed (sharpness={m_high['heatmap_sharpness']:.4f})")
            else:
                score.record_fail("Focused spot quality metrics below expected levels")

            flat_over = np.repeat((flat_map[:, :, None] * 255).astype(np.uint8), 3, axis=2)
            m_flat = checker.check_quality(orig, flat_map, flat_over)
            if m_flat["heatmap_sharpness"] == 0.0:
                score.record_pass("Flat activation map correctly identified zero sharpness")
            else:
                score.record_fail("Flat activation map incorrectly measured non-zero sharpness")

        except Exception as exc:
            score.record_fail(f"QualityEvaluator exception: {exc}")


__all__ = ["QualityEvaluator"]
