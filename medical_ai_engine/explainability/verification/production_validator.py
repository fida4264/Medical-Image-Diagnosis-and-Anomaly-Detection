"""
production_validator.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Master production readiness validator and verdict evaluator.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

from .verification_metrics import CategoryScore, VerificationMetrics

logger = logging.getLogger(__name__)


class ProductionValidator:
    """
    Evaluates production readiness criteria across all verification scores.

    Verdict Decision Rules
    ----------------------
    - **GO**: Overall score >= 90.0%, no critical failures in architecture, integration, or stress tests.
    - **NO GO**: Overall score < 90.0% or any critical domain failure.
    """

    MIN_OVERALL_SCORE_FOR_GO = 90.0

    def evaluate_verdict(self, metrics: VerificationMetrics) -> Dict[str, Any]:
        overall = metrics.overall_score_pct()
        categories = metrics.categories

        critical_domains = ["architecture", "integration", "gradcam", "resilience"]
        critical_failed = [
            dom for dom in critical_domains
            if categories.get(dom) and categories[dom].score_pct < 80.0
        ]

        is_go = (overall >= self.MIN_OVERALL_SCORE_FOR_GO) and (len(critical_failed) == 0)
        verdict = "GO" if is_go else "NO GO"

        res = {
            "verdict": verdict,
            "overall_score_pct": overall,
            "grade": self._grade(overall),
            "critical_domain_failures": critical_failed,
            "summary": f"Production Readiness Verdict: {verdict} (Overall Score: {overall:.1f}%)",
        }
        logger.info("[ProductionValidator] %s", res["summary"])
        return res

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 95.0:
            return "A+"
        if score >= 90.0:
            return "A"
        if score >= 80.0:
            return "B"
        if score >= 70.0:
            return "C"
        return "F"


__all__ = ["ProductionValidator"]
