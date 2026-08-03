"""
clinical_context.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Mutable context object that flows through the clinical pipeline.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ClinicalContext:
    """
    Mutable context that flows through all stages of the Clinical Explanation Engine.

    Fields are progressively populated as each pipeline stage executes.
    """

    # ── Identity ─────────────────────────────────────────────────────────────
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # ── Inputs ───────────────────────────────────────────────────────────────
    modality: str = ""
    disease_label: str = ""
    confidence_score: float = 0.0
    image_width: int = 0
    image_height: int = 0

    # ── Heatmap region inputs ─────────────────────────────────────────────────
    important_regions: List[Dict[str, Any]] = field(default_factory=list)
    visualization_metrics: Dict[str, float] = field(default_factory=dict)

    # ── Stage outputs (populated progressively) ───────────────────────────────
    classified_findings: List[Dict[str, Any]] = field(default_factory=list)
    severity_label: str = ""
    severity_score: float = 0.0
    region_descriptions: List[str] = field(default_factory=list)
    clinical_text: str = ""
    recommendations: List[str] = field(default_factory=list)
    report: Optional[Dict[str, Any]] = None

    # ── Timing ────────────────────────────────────────────────────────────────
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    total_time_ms: float = 0.0

    # ── Errors ────────────────────────────────────────────────────────────────
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # ── Helpers ──────────────────────────────────────────────────────────────

    @property
    def is_complete(self) -> bool:
        """True when the clinical report has been assembled."""
        return self.report is not None

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)

    def record_stage_time(self, stage: str, elapsed_ms: float) -> None:
        self.stage_timings_ms[stage] = round(elapsed_ms, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "created_at": self.created_at,
            "modality": self.modality,
            "disease_label": self.disease_label,
            "confidence_score": self.confidence_score,
            "severity_label": self.severity_label,
            "severity_score": self.severity_score,
            "clinical_text": self.clinical_text,
            "recommendations": self.recommendations,
            "region_descriptions": self.region_descriptions,
            "stage_timings_ms": self.stage_timings_ms,
            "total_time_ms": self.total_time_ms,
            "errors": self.errors,
            "warnings": self.warnings,
            "is_complete": self.is_complete,
        }


__all__ = ["ClinicalContext"]
