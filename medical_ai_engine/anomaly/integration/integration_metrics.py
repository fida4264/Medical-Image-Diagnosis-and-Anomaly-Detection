"""
integration_metrics.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Thread-safe performance latency metrics for Anomaly Integration.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnomalyTimings:
    """Detailed breakdown of latency measurements (ms)."""
    strategy_execution_ms: float = 0.0
    unknown_detection_ms: float = 0.0
    decision_fusion_ms: float = 0.0
    anomaly_total_latency_ms: float = 0.0
    overall_integration_ms: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "strategy_execution_ms": round(self.strategy_execution_ms, 3),
            "unknown_detection_ms": round(self.unknown_detection_ms, 3),
            "decision_fusion_ms": round(self.decision_fusion_ms, 3),
            "anomaly_total_latency_ms": round(self.anomaly_total_latency_ms, 3),
            "overall_integration_ms": round(self.overall_integration_ms, 3),
        }


class IntegrationMetrics:
    """
    Thread-safe performance metrics collector for Anomaly Integration.

    Measures:
    - Total anomaly latency
    - Strategy execution times
    - Decision fusion latency
    - Overall integration latency
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_runs = 0
        self._failed_runs = 0
        self._skipped_runs = 0
        self._accumulated_latency_ms = 0.0
        self._last_timings: AnomalyTimings = AnomalyTimings()

    def record_run(self, timings: AnomalyTimings, success: bool = True) -> None:
        with self._lock:
            self._total_requests += 1
            if success:
                self._successful_runs += 1
            else:
                self._failed_runs += 1
            self._accumulated_latency_ms += timings.overall_integration_ms
            self._last_timings = timings

    def record_skip(self) -> None:
        with self._lock:
            self._total_requests += 1
            self._skipped_runs += 1

    def get_summary(self) -> Dict[str, Any]:
        with self._lock:
            avg_latency = (
                self._accumulated_latency_ms / self._successful_runs
                if self._successful_runs > 0
                else 0.0
            )
            return {
                "total_requests": self._total_requests,
                "successful_runs": self._successful_runs,
                "failed_runs": self._failed_runs,
                "skipped_runs": self._skipped_runs,
                "avg_integration_latency_ms": round(avg_latency, 3),
                "last_timings": self._last_timings.as_dict(),
            }

    def reset(self) -> None:
        with self._lock:
            self._total_requests = 0
            self._successful_runs = 0
            self._failed_runs = 0
            self._skipped_runs = 0
            self._accumulated_latency_ms = 0.0
            self._last_timings = AnomalyTimings()


__all__ = ["IntegrationMetrics", "AnomalyTimings"]
