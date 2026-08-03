"""
integration_metrics.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Thread-safe performance metrics collector for the integration layer.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ExplainabilityTimings:
    """Per-request timing breakdown for one explainability execution."""
    request_id: str
    context_build_ms: float = 0.0
    gradcam_ms: float = 0.0
    heatmap_ms: float = 0.0
    overlay_ms: float = 0.0
    clinical_explanation_ms: float = 0.0
    result_attachment_ms: float = 0.0
    total_explainability_ms: float = 0.0
    success: bool = True
    modality: str = "unknown"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "context_build_ms": round(self.context_build_ms, 3),
            "gradcam_ms": round(self.gradcam_ms, 3),
            "heatmap_ms": round(self.heatmap_ms, 3),
            "overlay_ms": round(self.overlay_ms, 3),
            "clinical_explanation_ms": round(self.clinical_explanation_ms, 3),
            "result_attachment_ms": round(self.result_attachment_ms, 3),
            "total_explainability_ms": round(self.total_explainability_ms, 3),
            "success": self.success,
            "modality": self.modality,
        }


class IntegrationMetrics:
    """
    Thread-safe metrics store for Module 15.6 integration performance.

    Tracks:
      - Total requests processed
      - Successful / failed executions
      - Per-stage latency statistics
      - Modality distribution
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = 0
        self._successes = 0
        self._failures = 0
        self._skipped = 0
        self._timings: List[ExplainabilityTimings] = []
        self._modality_counts: Dict[str, int] = {}

    # ── Recording ─────────────────────────────────────────────────────────────

    def record(self, timings: ExplainabilityTimings) -> None:
        with self._lock:
            self._total += 1
            self._timings.append(timings)
            if timings.success:
                self._successes += 1
            else:
                self._failures += 1
            mod = timings.modality or "unknown"
            self._modality_counts[mod] = self._modality_counts.get(mod, 0) + 1

    def record_skip(self) -> None:
        with self._lock:
            self._total += 1
            self._skipped += 1

    # ── Queries ───────────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            timings = list(self._timings)

        if not timings:
            avg_total = 0.0
            avg_gradcam = 0.0
            avg_clinical = 0.0
        else:
            avg_total = sum(t.total_explainability_ms for t in timings) / len(timings)
            avg_gradcam = sum(t.gradcam_ms for t in timings) / len(timings)
            avg_clinical = sum(t.clinical_explanation_ms for t in timings) / len(timings)

        with self._lock:
            return {
                "total_processed": self._total,
                "successful": self._successes,
                "failed": self._failures,
                "skipped": self._skipped,
                "success_rate": round(self._successes / max(self._total - self._skipped, 1), 4),
                "avg_total_explainability_ms": round(avg_total, 3),
                "avg_gradcam_ms": round(avg_gradcam, 3),
                "avg_clinical_explanation_ms": round(avg_clinical, 3),
                "modality_distribution": dict(self._modality_counts),
            }

    def reset(self) -> None:
        with self._lock:
            self._total = 0
            self._successes = 0
            self._failures = 0
            self._skipped = 0
            self._timings.clear()
            self._modality_counts.clear()


__all__ = ["IntegrationMetrics", "ExplainabilityTimings"]
