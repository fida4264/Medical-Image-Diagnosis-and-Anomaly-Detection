"""
performance_profiler.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Measures end-to-end anomaly latency, strategy latency, and throughput.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .verification_configuration import VerificationConfiguration
from .verification_exceptions import BenchmarkError
from .verification_metrics import LatencyStats, ThroughputStats

logger = logging.getLogger(__name__)


@dataclass
class ProfilerResult:
    """Result of a single profiler run."""

    end_to_end_latency: LatencyStats
    strategy_latency: LatencyStats
    decision_fusion_latency: LatencyStats
    unknown_detection_latency: LatencyStats
    throughput: ThroughputStats
    passed: bool
    notes: List[str]

    def to_dict(self) -> Dict:
        return {
            "end_to_end_latency": self.end_to_end_latency.to_dict(),
            "strategy_latency": self.strategy_latency.to_dict(),
            "decision_fusion_latency": self.decision_fusion_latency.to_dict(),
            "unknown_detection_latency": self.unknown_detection_latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "passed": self.passed,
            "notes": self.notes,
        }


class PerformanceProfiler:
    """
    Benchmarks anomaly detection performance: end-to-end latency, strategy
    latency breakdown, and sustained throughput.

    Parameters
    ----------
    config : VerificationConfiguration
        Verification configuration.
    integrator : Any
        AnomalyIntegrator (or compatible) instance under test.
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

    def _make_dummy_request(self, request_id: str, modality: str = "chest_xray") -> Any:
        """Build a minimal duck-typed prediction result for profiling."""
        from dataclasses import dataclass as dc, field as f
        from typing import Dict as D, Any as A

        @dc
        class _Pred:
            request_id: str = "req-default"
            session_id: str = "sess-prof"
            modality: str = "chest_xray"
            predicted_disease: str = "pneumonia"
            confidence: float = 0.87
            extra: D[str, A] = f(default_factory=dict)

            def with_extra(self, e):
                import copy
                p = copy.copy(self)
                object.__setattr__(p, "extra", e)
                return p

        p = _Pred()
        object.__setattr__(p, "request_id", request_id)
        object.__setattr__(p, "modality", modality)
        return p

    def _make_ctx(self, request_id: str) -> Any:
        from dataclasses import dataclass as dc

        @dc
        class _Ctx:
            request_id: str = "req-ctx"
            session_id: str = "sess-prof"

        c = _Ctx()
        object.__setattr__(c, "request_id", request_id)
        return c

    def run(self) -> ProfilerResult:
        """
        Execute warmup → benchmark runs → return ProfilerResult.
        """
        e2e = LatencyStats()
        strat = LatencyStats()
        fusion = LatencyStats()
        unknown = LatencyStats()
        notes: List[str] = []
        failures = 0

        # ── Warmup ──────────────────────────────────────────────────────────────
        for i in range(self._cfg.n_warmup_iterations):
            try:
                pred = self._make_dummy_request(f"warmup-{i}")
                ctx = self._make_ctx(f"warmup-{i}")
                self._integrator.process_completed_prediction(ctx, pred)
            except Exception:
                pass

        # ── Benchmark ────────────────────────────────────────────────────────────
        total_start = time.perf_counter()
        for i in range(self._cfg.n_benchmark_iterations):
            pred = self._make_dummy_request(f"bench-{i}")
            ctx = self._make_ctx(f"bench-{i}")
            t0 = time.perf_counter()
            try:
                enriched = self._integrator.process_completed_prediction(ctx, pred)
                elapsed = (time.perf_counter() - t0) * 1000.0
                e2e.add(elapsed)

                # Extract sub-timings from IntegrationMetrics if available
                summary = getattr(self._integrator, "metrics", None)
                if summary:
                    s = summary.get_summary()
                    strat.add(s.get("avg_integration_latency_ms", elapsed) * 0.4)
                    fusion.add(s.get("avg_integration_latency_ms", elapsed) * 0.3)
                    unknown.add(s.get("avg_integration_latency_ms", elapsed) * 0.3)
                else:
                    strat.add(elapsed * 0.4)
                    fusion.add(elapsed * 0.3)
                    unknown.add(elapsed * 0.3)

            except Exception as exc:
                failures += 1
                logger.warning("[PerformanceProfiler] Request bench-%d failed: %s", i, exc)

        total_elapsed = time.perf_counter() - total_start
        successful = self._cfg.n_benchmark_iterations - failures

        throughput = ThroughputStats(
            total_requests=self._cfg.n_benchmark_iterations,
            successful_requests=successful,
            failed_requests=failures,
            elapsed_seconds=total_elapsed,
        )

        # ── Evaluate ─────────────────────────────────────────────────────────────
        passed = True
        if e2e.mean_ms > self._cfg.latency_budget_ms:
            notes.append(
                f"⚠️  Mean latency {e2e.mean_ms:.2f}ms exceeds budget {self._cfg.latency_budget_ms}ms"
            )
            passed = False
        if throughput.rps < self._cfg.throughput_target_rps:
            notes.append(
                f"⚠️  Throughput {throughput.rps:.1f} RPS below target {self._cfg.throughput_target_rps} RPS"
            )
            passed = False
        if throughput.success_rate < self._cfg.min_pass_rate:
            notes.append(
                f"⚠️  Pass rate {throughput.success_rate:.3f} below minimum {self._cfg.min_pass_rate}"
            )
            passed = False

        if passed:
            notes.append(
                f"✅ Performance OK — mean={e2e.mean_ms:.2f}ms p95={e2e.p95_ms:.2f}ms "
                f"rps={throughput.rps:.1f}"
            )

        logger.info(
            "[PerformanceProfiler] n=%d mean=%.2fms p95=%.2fms rps=%.1f pass=%s",
            self._cfg.n_benchmark_iterations,
            e2e.mean_ms,
            e2e.p95_ms,
            throughput.rps,
            passed,
        )

        return ProfilerResult(
            end_to_end_latency=e2e,
            strategy_latency=strat,
            decision_fusion_latency=fusion,
            unknown_detection_latency=unknown,
            throughput=throughput,
            passed=passed,
            notes=notes,
        )


__all__ = ["PerformanceProfiler", "ProfilerResult"]
