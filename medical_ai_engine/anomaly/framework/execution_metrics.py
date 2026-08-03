"""
execution_metrics.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Thread-safe execution metrics for the framework.
"""

from __future__ import annotations

import statistics
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StrategyMetricsSnapshot:
    """Metrics snapshot for an individual strategy."""
    strategy_key: str
    total_invocations: int = 0
    successful_invocations: int = 0
    failed_invocations: int = 0
    cached_hits: int = 0
    latencies_ms: List[float] = field(default_factory=list)

    def record_execution(self, latency_ms: float, success: bool, cached: bool = False) -> None:
        self.total_invocations += 1
        if cached:
            self.cached_hits += 1
        if success:
            self.successful_invocations += 1
        else:
            self.failed_invocations += 1
        self.latencies_ms.append(latency_ms)

    def summary(self) -> Dict[str, Any]:
        mean_ms = statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0
        return {
            "strategy_key": self.strategy_key,
            "total_invocations": self.total_invocations,
            "successful_invocations": self.successful_invocations,
            "failed_invocations": self.failed_invocations,
            "cached_hits": self.cached_hits,
            "mean_latency_ms": round(mean_ms, 3),
        }


class ExecutionMetrics:
    """
    Thread-safe framework-wide metrics collector.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._strategy_metrics: Dict[str, StrategyMetricsSnapshot] = {}
        self._total_pipeline_runs: int = 0
        self._failed_pipeline_runs: int = 0

    def record_strategy_run(
        self, strategy_key: str, latency_ms: float, success: bool, cached: bool = False
    ) -> None:
        with self._lock:
            if strategy_key not in self._strategy_metrics:
                self._strategy_metrics[strategy_key] = StrategyMetricsSnapshot(
                    strategy_key=strategy_key
                )
            self._strategy_metrics[strategy_key].record_execution(
                latency_ms=latency_ms, success=success, cached=cached
            )

    def record_pipeline_run(self, success: bool) -> None:
        with self._lock:
            self._total_pipeline_runs += 1
            if not success:
                self._failed_pipeline_runs += 1

    def get_strategy_metrics(self, strategy_key: str) -> Dict[str, Any]:
        with self._lock:
            if strategy_key in self._strategy_metrics:
                return self._strategy_metrics[strategy_key].summary()
            return {"strategy_key": strategy_key, "total_invocations": 0}

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "total_pipeline_runs": self._total_pipeline_runs,
                "failed_pipeline_runs": self._failed_pipeline_runs,
                "strategies": {
                    key: m.summary() for key, m in self._strategy_metrics.items()
                },
            }


__all__ = ["ExecutionMetrics", "StrategyMetricsSnapshot"]
