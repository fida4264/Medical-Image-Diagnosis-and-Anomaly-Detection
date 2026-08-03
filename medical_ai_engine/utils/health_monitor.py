"""
Medical AI Engine — Health Monitor

Monitors engine health status, system memory, active threads, and state consistency.
"""

from __future__ import annotations

from datetime import datetime, timezone
import psutil
import threading
from typing import Any, Dict, Optional


class EngineHealthMonitor:
    """
    Health monitoring system for MedicalAIEngine.
    Evaluates memory usage, active threads, process state, and component health.
    """

    def __init__(self, engine_name: str = "MedicalAIEngineCore") -> None:
        self.engine_name = engine_name
        self._lock = threading.Lock()

    def check_health(
        self,
        current_state: str,
        active_sessions_count: int,
        statistics_snapshot: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Perform a health assessment.

        Returns:
            Dictionary containing health metrics, status ("HEALTHY", "DEGRADED", "UNHEALTHY"),
            memory usage, CPU usage, thread count, and timestamps.
        """
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_percent = process.memory_percent()
        cpu_percent = process.cpu_percent(interval=None)
        thread_count = process.num_threads()

        status = "HEALTHY"
        issues = []

        if mem_percent > 85.0:
            status = "DEGRADED"
            issues.append(f"High memory usage: {mem_percent:.1f}%")

        if current_state == "FAILED":
            status = "UNHEALTHY"
            issues.append("Engine state is FAILED")

        return {
            "engine_name": self.engine_name,
            "status": status,
            "current_state": current_state,
            "active_sessions": active_sessions_count,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "metrics": {
                "memory_rss_mb": round(mem_info.rss / (1024 * 1024), 2),
                "memory_percent": round(mem_percent, 2),
                "cpu_percent": round(cpu_percent, 2),
                "active_threads": thread_count,
            },
            "issues": issues,
            "statistics": statistics_snapshot or {},
        }
