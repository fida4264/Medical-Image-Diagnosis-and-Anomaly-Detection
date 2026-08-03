"""
Medical AI Engine — Explainability Result

Module 15.1: Explainability Framework Architecture

Immutable standardized output object representing the final generated
explainability result.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExplainabilityResult:
    """
    Immutable dataclass representing the standardized output of the Explainability Engine.
    """

    # ── Identity & Request ────────────────────────────────────────────────────
    request_id: str
    session_id: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ── Target & Method ───────────────────────────────────────────────────────
    method: str = "gradcam"
    modality: str = "unknown"
    predicted_class: str = "unknown"
    confidence: float = 0.0
    target_layer: str = "unknown"

    # ── Artifact Paths ────────────────────────────────────────────────────────
    heatmap_path: Optional[str] = None
    overlay_path: Optional[str] = None
    activation_map_path: Optional[str] = None

    # ── Clinical Region Analysis ──────────────────────────────────────────────
    important_regions: List[Dict[str, Any]] = field(default_factory=list)
    clinical_summary: str = ""

    # ── Quality & Performance ─────────────────────────────────────────────────
    quality_score: float = 100.0             # Heatmap clarity / localization quality score
    generation_time_ms: float = 0.0
    status: str = "SUCCESS"                  # SUCCESS, FAILED, WARNING

    # ── Audit Trail & Extension ───────────────────────────────────────────────
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_success(self) -> bool:
        return self.status in ("SUCCESS", "COMPLETED_WITH_WARNINGS")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize ExplainabilityResult to dictionary."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize ExplainabilityResult to JSON string."""
        return json.dumps(self.to_dict(), default=str)

    def to_pretty_json(self) -> str:
        """Serialize ExplainabilityResult to indented human-readable JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def __repr__(self) -> str:
        return (
            f"ExplainabilityResult("
            f"request_id={self.request_id!r}, "
            f"method={self.method!r}, "
            f"predicted_class={self.predicted_class!r}, "
            f"status={self.status!r})"
        )
