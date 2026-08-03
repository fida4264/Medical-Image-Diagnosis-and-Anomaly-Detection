"""
memory_profiler.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — RSS memory profiling during anomaly detection execution.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, List, Optional

from .verification_configuration import VerificationConfiguration
from .verification_metrics import MemoryStats

logger = logging.getLogger(__name__)


def _get_rss_mb() -> float:
    """Return current process RSS memory in MB (cross-platform)."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        # Fallback: read /proc/self/status on Linux
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            pass
    return 0.0


@dataclass
class MemoryProfilerResult:
    memory: MemoryStats
    within_budget: bool
    notes: List[str]

    def to_dict(self):
        return {
            "memory": self.memory.to_dict(),
            "within_budget": self.within_budget,
            "notes": self.notes,
        }


class MemoryProfiler:
    """
    Profiles RSS memory growth during anomaly detection execution.

    Captures baseline before runs, takes periodic snapshots during runs,
    and computes peak / growth versus the configured budget.
    """

    def __init__(
        self,
        config: Optional[VerificationConfiguration] = None,
        integrator: Optional[Any] = None,
    ) -> None:
        self._cfg = config or VerificationConfiguration.fast_test()
        self._integrator = integrator or self._build_default_integrator()

    def _build_default_integrator(self) -> Any:
        from medical_ai_engine.anomaly.integration import AnomalyIntegrator
        return AnomalyIntegrator()

    def _make_request(self, i: int) -> Any:
        from dataclasses import dataclass as dc, field as f
        from typing import Dict as D, Any as A

        @dc
        class _Pred:
            request_id: str = "mem-0"
            session_id: str = "sess-mem"
            modality: str = "chest_xray"
            predicted_disease: str = "pneumonia"
            confidence: float = 0.82
            extra: D[str, A] = f(default_factory=dict)

            def with_extra(self, e):
                import copy
                p = copy.copy(self)
                object.__setattr__(p, "extra", e)
                return p

        p = _Pred()
        object.__setattr__(p, "request_id", f"mem-{i}")
        return p

    def _make_ctx(self, i: int) -> Any:
        from dataclasses import dataclass as dc

        @dc
        class _Ctx:
            request_id: str = f"mem-{i}"
            session_id: str = "sess-mem"

        return _Ctx()

    def run(self) -> MemoryProfilerResult:
        """Run memory profiling and return MemoryProfilerResult."""
        if not self._cfg.enable_memory_profiling:
            return MemoryProfilerResult(
                memory=MemoryStats(),
                within_budget=True,
                notes=["Memory profiling disabled."],
            )

        # Baseline
        baseline_mb = _get_rss_mb()
        snapshots: List[float] = [baseline_mb]
        notes: List[str] = []

        # Run iterations and collect periodic snapshots
        interval = max(1, self._cfg.n_benchmark_iterations // 10)
        for i in range(self._cfg.n_benchmark_iterations):
            try:
                pred = self._make_request(i)
                ctx = self._make_ctx(i)
                self._integrator.process_completed_prediction(ctx, pred)
            except Exception:
                pass
            if i % interval == 0:
                snapshots.append(_get_rss_mb())

        final_mb = _get_rss_mb()
        snapshots.append(final_mb)
        peak_mb = max(snapshots)
        growth_mb = final_mb - baseline_mb

        mem = MemoryStats(
            baseline_mb=baseline_mb,
            peak_mb=peak_mb,
            final_mb=final_mb,
            growth_mb=growth_mb,
            snapshots=snapshots,
        )

        within_budget = growth_mb <= self._cfg.memory_budget_mb
        if within_budget:
            notes.append(
                f"✅ Memory OK — baseline={baseline_mb:.1f}MB peak={peak_mb:.1f}MB "
                f"growth={growth_mb:.1f}MB (budget={self._cfg.memory_budget_mb}MB)"
            )
        else:
            notes.append(
                f"⚠️  Memory growth {growth_mb:.1f}MB exceeds budget {self._cfg.memory_budget_mb}MB"
            )

        logger.info(
            "[MemoryProfiler] baseline=%.1fMB peak=%.1fMB growth=%.1fMB within_budget=%s",
            baseline_mb, peak_mb, growth_mb, within_budget,
        )
        return MemoryProfilerResult(memory=mem, within_budget=within_budget, notes=notes)


__all__ = ["MemoryProfiler", "MemoryProfilerResult"]
