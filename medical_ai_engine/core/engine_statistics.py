"""
Medical AI Engine — EngineStatistics

Thread-safe statistics tracking for the MedicalAIEngine Core.
Tracks execution count, latency totals/averages, active/completed/failed sessions,
uptime, and average initialization time.
"""

from __future__ import annotations

from datetime import datetime, timezone
import threading
import time
from typing import Any, Dict


class EngineStatistics:
    """
    Thread-safe operational statistics collector for MedicalAIEngine.
    Uses RLock to protect counter updates across concurrent prediction threads.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()

        self._start_time_monotonic = time.monotonic()
        self._start_time_utc = datetime.now(tz=timezone.utc).isoformat()

        # Counter metrics
        self._total_executions: int = 0
        self._active_sessions: int = 0
        self._completed_sessions: int = 0
        self._failed_sessions: int = 0
        self._cancelled_sessions: int = 0

        # Latency metrics (seconds)
        self._total_execution_time_seconds: float = 0.0
        self._total_init_time_seconds: float = 0.0
        self._init_count: int = 0

        self._min_execution_time_ms: float = float("inf")
        self._max_execution_time_ms: float = 0.0

    def record_engine_init(self, duration_seconds: float) -> None:
        """Record an engine initialization event and its duration."""
        with self._lock:
            self._init_count += 1
            self._total_init_time_seconds += duration_seconds

    def session_started(self) -> None:
        """Increment active sessions and total executions counter."""
        with self._lock:
            self._total_executions += 1
            self._active_sessions += 1

    def session_completed(self, duration_ms: float) -> None:
        """Record a successful session completion with latency."""
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1
            self._completed_sessions += 1

            duration_s = duration_ms / 1000.0
            self._total_execution_time_seconds += duration_s

            if duration_ms < self._min_execution_time_ms:
                self._min_execution_time_ms = duration_ms
            if duration_ms > self._max_execution_time_ms:
                self._max_execution_time_ms = duration_ms

    def session_failed(self) -> None:
        """Record a session failure."""
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1
            self._failed_sessions += 1

    def session_cancelled(self) -> None:
        """Record a session cancellation."""
        with self._lock:
            if self._active_sessions > 0:
                self._active_sessions -= 1
            self._cancelled_sessions += 1

    def uptime_seconds(self) -> float:
        """Return total engine uptime in seconds."""
        return round(time.monotonic() - self._start_time_monotonic, 2)

    def average_execution_time_ms(self) -> float:
        """Return average execution time in milliseconds."""
        with self._lock:
            if self._completed_sessions == 0:
                return 0.0
            total_ms = self._total_execution_time_seconds * 1000.0
            return round(total_ms / self._completed_sessions, 3)

    def average_initialization_time_ms(self) -> float:
        """Return average initialization time in milliseconds."""
        with self._lock:
            if self._init_count == 0:
                return 0.0
            total_ms = self._total_init_time_seconds * 1000.0
            return round(total_ms / self._init_count, 3)

    def success_rate(self) -> float:
        """Return session success percentage (0.0 to 100.0)."""
        with self._lock:
            total_finished = self._completed_sessions + self._failed_sessions + self._cancelled_sessions
            if total_finished == 0:
                return 100.0
            return round((self._completed_sessions / total_finished) * 100.0, 2)

    def reset(self) -> None:
        """Reset all counters and latency metrics."""
        with self._lock:
            self._start_time_monotonic = time.monotonic()
            self._start_time_utc = datetime.now(tz=timezone.utc).isoformat()
            self._total_executions = 0
            self._active_sessions = 0
            self._completed_sessions = 0
            self._failed_sessions = 0
            self._cancelled_sessions = 0
            self._total_execution_time_seconds = 0.0
            self._total_init_time_seconds = 0.0
            self._init_count = 0
            self._min_execution_time_ms = float("inf")
            self._max_execution_time_ms = 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Return a complete dictionary snapshot of all current metrics."""
        with self._lock:
            min_ms = self._min_execution_time_ms if self._min_execution_time_ms != float("inf") else 0.0
            return {
                "start_time_utc": self._start_time_utc,
                "uptime_seconds": self.uptime_seconds(),
                "total_executions": self._total_executions,
                "active_sessions": self._active_sessions,
                "completed_sessions": self._completed_sessions,
                "failed_sessions": self._failed_sessions,
                "cancelled_sessions": self._cancelled_sessions,
                "success_rate_percent": self.success_rate(),
                "average_execution_time_ms": self.average_execution_time_ms(),
                "min_execution_time_ms": round(min_ms, 3),
                "max_execution_time_ms": round(self._max_execution_time_ms, 3),
                "average_init_time_ms": self.average_initialization_time_ms(),
                "total_init_count": self._init_count,
            }

    def __repr__(self) -> str:
        return (
            f"EngineStatistics("
            f"executions={self._total_executions}, "
            f"active={self._active_sessions}, "
            f"completed={self._completed_sessions}, "
            f"failed={self._failed_sessions}, "
            f"avg_ms={self.average_execution_time_ms():.2f}ms)"
        )
