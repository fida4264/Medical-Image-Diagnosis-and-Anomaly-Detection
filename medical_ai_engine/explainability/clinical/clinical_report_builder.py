"""
clinical_report_builder.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Assembles the final structured clinical report dict.

Responsibility
--------------
Collects all stage outputs from ClinicalContext and assembles a structured,
JSON-serialisable report dictionary including metadata, findings, severity,
clinical text, recommendations, quality metrics, and timing.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import ReportBuildError

logger = logging.getLogger(__name__)

_REPORT_SCHEMA_VERSION = "1.0.0"


class ClinicalReportBuilder:
    """
    Assembles and optionally persists the final clinical report.

    The report structure
    --------------------
    {
        "schema_version":   str,
        "generated_at":     ISO-8601 timestamp,
        "request_id":       str,
        "session_id":       str,
        "modality":         str,
        "disease_label":    str,
        "confidence_score": float,
        "severity": {
            "label":  str,
            "score":  float,
        },
        "clinical_impression": str,         # Full multi-sentence paragraph
        "findings": [                        # Classified finding dicts
            {anatomy, pathology, peak_score, area_fraction, significance, …}
        ],
        "region_descriptions": [str],
        "recommendations":    [str],
        "visualization_metrics": {str: float},
        "performance": {
            "stage_timings_ms": {str: float},
            "total_time_ms":    float,
        },
        "warnings": [str],
        "configuration": {…},
    }
    """

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def build(
        self,
        ctx: ClinicalContext,
        export_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Build the final report from *ctx*, populate *ctx.report*, and optionally
        write it to *export_path* (as JSON). Returns the report dict.
        """
        t0 = time.perf_counter()
        try:
            report = self._assemble(ctx)
            ctx.report = report
            elapsed = (time.perf_counter() - t0) * 1000
            ctx.record_stage_time("report_build", elapsed)
            ctx.total_time_ms = sum(ctx.stage_timings_ms.values())

            # Update total_time inside the report
            report["performance"]["total_time_ms"] = round(ctx.total_time_ms, 3)

            if export_path:
                self._export(report, export_path)

            logger.info(
                "ClinicalReportBuilder: report assembled for request=%s in %.1f ms",
                ctx.request_id,
                ctx.total_time_ms,
            )
            return report
        except Exception as exc:
            msg = f"ClinicalReportBuilder failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise ReportBuildError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _assemble(self, ctx: ClinicalContext) -> Dict[str, Any]:
        return {
            "schema_version": _REPORT_SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "request_id": ctx.request_id,
            "session_id": ctx.session_id,
            "modality": ctx.modality,
            "disease_label": ctx.disease_label,
            "confidence_score": round(ctx.confidence_score, 4),
            "severity": {
                "label": ctx.severity_label,
                "score": ctx.severity_score,
            },
            "clinical_impression": ctx.clinical_text,
            "findings": ctx.classified_findings,
            "region_descriptions": ctx.region_descriptions,
            "recommendations": ctx.recommendations,
            "visualization_metrics": ctx.visualization_metrics,
            "performance": {
                "stage_timings_ms": dict(ctx.stage_timings_ms),
                "total_time_ms": 0.0,  # filled after build
            },
            "warnings": list(ctx.warnings),
            "configuration": self._config.to_dict(),
        }

    @staticmethod
    def _export(report: Dict[str, Any], path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False, default=str)
        logger.debug("ClinicalReportBuilder: report written to %s", path)


__all__ = ["ClinicalReportBuilder"]
