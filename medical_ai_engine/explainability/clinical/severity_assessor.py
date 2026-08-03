"""
severity_assessor.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Assesses clinical severity from confidence scores and activation metrics.

Responsibility
--------------
Maps (confidence_score, visualization_metrics, classified_findings) → a severity
label and numeric severity score, populating ClinicalContext.severity_label and
ClinicalContext.severity_score.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import SeverityAssessmentError

logger = logging.getLogger(__name__)


class SeverityAssessor:
    """
    Computes a composite severity score and label for an AI prediction.

    Composite score formula
    -----------------------
    severity_score = 0.50 * confidence_score
                   + 0.30 * peak_activation_mean
                   + 0.20 * coverage_ratio

    where coverage_ratio = visualization_metrics.get("coverage_ratio", 0.0)
    and   peak_activation_mean = mean of all finding peak_scores.
    """

    _WEIGHT_CONFIDENCE   = 0.50
    _WEIGHT_PEAK         = 0.30
    _WEIGHT_COVERAGE     = 0.20

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def assess(self, ctx: ClinicalContext) -> str:
        """
        Assess severity from *ctx* and populate ctx.severity_label / severity_score.
        Returns the severity label string.
        """
        t0 = time.perf_counter()
        try:
            label, score = self._compute(ctx)
            ctx.severity_label = label
            ctx.severity_score = score
            ctx.record_stage_time(
                "severity_assessment",
                (time.perf_counter() - t0) * 1000,
            )
            logger.debug(
                "SeverityAssessor: modality=%s disease=%s → %s (%.4f)",
                ctx.modality, ctx.disease_label, label, score,
            )
            return label
        except Exception as exc:
            msg = f"SeverityAssessor failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise SeverityAssessmentError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, ctx: ClinicalContext) -> tuple[str, float]:
        confidence = float(ctx.confidence_score or 0.0)
        coverage   = float(
            ctx.visualization_metrics.get("coverage_ratio", 0.0)
        )
        findings: List[Dict[str, Any]] = ctx.classified_findings or []
        if findings:
            peak_mean = sum(f.get("peak_score", 0.0) for f in findings) / len(findings)
        else:
            peak_mean = confidence

        score = (
            self._WEIGHT_CONFIDENCE * confidence
            + self._WEIGHT_PEAK     * peak_mean
            + self._WEIGHT_COVERAGE * coverage
        )
        score = min(1.0, max(0.0, score))
        label = self._config.classify_severity(score)
        return label, round(score, 4)


__all__ = ["SeverityAssessor"]
