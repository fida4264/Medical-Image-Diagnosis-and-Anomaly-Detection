"""
Medical AI Engine — Integration Metrics

Records, aggregates, and summarizes latency and throughput performance metrics
across end-to-end prediction workflows.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("IntegrationMetrics")


class IntegrationMetrics:
    """
    Thread-safe performance metrics collector for end-to-end workflows.
    Tracks:
      - Total execution time
      - Stage timing breakdown
      - Routing latency
      - Adapter latency
      - Response construction latency
      - Total workflow count, successful, and failed counts
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self.total_workflows: int = 0
        self.successful_workflows: int = 0
        self.failed_workflows: int = 0

        self.total_execution_times_ms: List[float] = []
        self.routing_latencies_ms: List[float] = []
        self.adapter_latencies_ms: List[float] = []
        self.response_latencies_ms: List[float] = []
        self.modality_counts: Dict[str, int] = {}

    def record_workflow(
        self,
        total_time_ms: float,
        routing_latency_ms: float = 0.0,
        adapter_latency_ms: float = 0.0,
        response_latency_ms: float = 0.0,
        modality: str = "unknown",
        success: bool = True,
    ) -> None:
        """Record metrics for a completed or failed prediction workflow."""
        with self._lock:
            self.total_workflows += 1
            if success:
                self.successful_workflows += 1
            else:
                self.failed_workflows += 1

            self.total_execution_times_ms.append(round(total_time_ms, 3))
            if routing_latency_ms > 0:
                self.routing_latencies_ms.append(round(routing_latency_ms, 3))
            if adapter_latency_ms > 0:
                self.adapter_latencies_ms.append(round(adapter_latency_ms, 3))
            if response_latency_ms > 0:
                self.response_latencies_ms.append(round(response_latency_ms, 3))

            self.modality_counts[modality] = self.modality_counts.get(modality, 0) + 1

    def reset(self) -> None:
        """Reset all counters and recorded latency samples."""
        with self._lock:
            self.total_workflows = 0
            self.successful_workflows = 0
            self.failed_workflows = 0
            self.total_execution_times_ms.clear()
            self.routing_latencies_ms.clear()
            self.adapter_latencies_ms.clear()
            self.response_latencies_ms.clear()
            self.modality_counts.clear()

    def get_summary(self) -> Dict[str, Any]:
        """Compute statistical summary dictionary."""
        with self._lock:
            avg_total = (
                sum(self.total_execution_times_ms) / len(self.total_execution_times_ms)
                if self.total_execution_times_ms
                else 0.0
            )
            avg_routing = (
                sum(self.routing_latencies_ms) / len(self.routing_latencies_ms)
                if self.routing_latencies_ms
                else 0.0
            )
            avg_adapter = (
                sum(self.adapter_latencies_ms) / len(self.adapter_latencies_ms)
                if self.adapter_latencies_ms
                else 0.0
            )
            avg_response = (
                sum(self.response_latencies_ms) / len(self.response_latencies_ms)
                if self.response_latencies_ms
                else 0.0
            )

            success_rate = (
                (self.successful_workflows / self.total_workflows) * 100.0
                if self.total_workflows > 0
                else 0.0
            )

            return {
                "total_workflows": self.total_workflows,
                "successful_workflows": self.successful_workflows,
                "failed_workflows": self.failed_workflows,
                "success_rate_pct": round(success_rate, 2),
                "avg_total_execution_ms": round(avg_total, 3),
                "min_total_execution_ms": round(min(self.total_execution_times_ms), 3) if self.total_execution_times_ms else 0.0,
                "max_total_execution_ms": round(max(self.total_execution_times_ms), 3) if self.total_execution_times_ms else 0.0,
                "avg_routing_latency_ms": round(avg_routing, 3),
                "avg_adapter_latency_ms": round(avg_adapter, 3),
                "avg_response_construction_latency_ms": round(avg_response, 3),
                "workflows_per_modality": dict(self.modality_counts),
            }

    def snapshot(self) -> Dict[str, Any]:
        """Return a snapshot dictionary of metrics (alias for get_summary)."""
        return self.get_summary()
