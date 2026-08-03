"""
execution_context.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Execution context wrapper for pipeline execution state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext


@dataclass
class FrameworkExecutionContext:
    """
    Framework-level wrapper around AnomalyContext to track pipeline execution state.

    Attributes
    ----------
    context : AnomalyContext
        The core domain context passed through anomaly detection.
    active_strategy_key : str
        The strategy currently executing or selected.
    cache_hit : bool
        True if the execution result was served from cache.
    start_time : float
        Perf_counter timestamp at execution start.
    stage_timings_ms : Dict[str, float]
        Latency breakdown by execution stage.
    metadata : Dict[str, Any]
        Framework internal metadata.
    """

    context: AnomalyContext
    active_strategy_key: str = "none"
    cache_hit: bool = False
    start_time: float = field(default_factory=time.perf_counter)
    stage_timings_ms: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_stage_timing(self, stage_name: str, duration_ms: float) -> None:
        self.stage_timings_ms[stage_name] = round(duration_ms, 3)

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.start_time) * 1000.0


__all__ = ["FrameworkExecutionContext"]
