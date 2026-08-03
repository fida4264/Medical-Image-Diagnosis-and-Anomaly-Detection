"""
benchmark_runner.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Structured benchmark suite: per-strategy, per-modality, pipeline.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .verification_configuration import VerificationConfiguration
from .verification_metrics import LatencyStats

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkRow:
    """Single row in a benchmark table."""
    scenario: str
    mean_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    iterations: int

    def to_dict(self) -> Dict:
        return {
            "scenario": self.scenario,
            "mean_ms": round(self.mean_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "iterations": self.iterations,
        }


@dataclass
class BenchmarkResult:
    rows: List[BenchmarkRow]

    def to_dict(self) -> Dict:
        return {"benchmarks": [r.to_dict() for r in self.rows]}

    def as_markdown_table(self) -> str:
        header = (
            "| Scenario | Mean (ms) | P95 (ms) | P99 (ms) | Min (ms) | Max (ms) | N |\n"
            "|----------|-----------|----------|----------|----------|----------|---|\n"
        )
        rows = "\n".join(
            f"| {r.scenario} | {r.mean_ms:.3f} | {r.p95_ms:.3f} | {r.p99_ms:.3f} | "
            f"{r.min_ms:.3f} | {r.max_ms:.3f} | {r.iterations} |"
            for r in self.rows
        )
        return header + rows


class BenchmarkRunner:
    """
    Runs structured benchmarks for individual anomaly sub-components and the
    full end-to-end pipeline, grouped by scenario.

    Scenarios
    ---------
    - full_pipeline        : end-to-end AnomalyIntegrator.process_completed_prediction
    - confidence_strategy  : ConfidenceThresholdStrategy.detect only
    - mahalanobis_strategy : MahalanobisOODStrategy.detect only
    - unknown_detector     : UnknownDiseaseDetector.analyze only
    - decision_fusion      : DecisionEngine.evaluate_signals only
    - per_modality_{name}  : full pipeline per modality
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

    def _make_ctx(self, request_id: str, modality: str = "chest_xray",
                  confidence: float = 0.88) -> Any:
        from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
        return AnomalyContext(
            request_id=request_id,
            session_id="sess-bench",
            modality=modality,
            confidence=confidence,
            predicted_class="pneumonia",
        )

    def _make_engine_ctx(self, request_id: str) -> Any:
        from dataclasses import dataclass as dc

        @dc
        class _Ctx:
            request_id: str = "req-ctx"
            session_id: str = "sess-bench"

        c = _Ctx()
        object.__setattr__(c, "request_id", request_id)
        return c

    def _make_pred(self, request_id: str, modality: str = "chest_xray",
                   confidence: float = 0.88) -> Any:
        from dataclasses import dataclass as dc, field as f
        from typing import Dict as D, Any as A

        @dc
        class _Pred:
            request_id: str = "req-bench"
            session_id: str = "sess-bench"
            modality: str = "chest_xray"
            predicted_disease: str = "pneumonia"
            confidence: float = 0.88
            extra: D[str, A] = f(default_factory=dict)

            def with_extra(self, e):
                import copy
                p = copy.copy(self)
                object.__setattr__(p, "extra", e)
                return p

        p = _Pred()
        object.__setattr__(p, "request_id", request_id)
        object.__setattr__(p, "modality", modality)
        object.__setattr__(p, "confidence", confidence)
        return p

    def _make_engine_ctx(self, request_id: str) -> Any:
        from dataclasses import dataclass as dc

        @dc
        class _Ctx:
            request_id: str = request_id
            session_id: str = "sess-bench"

        return _Ctx(request_id=request_id)

    def _time_scenario(self, name: str, fn) -> BenchmarkRow:
        """Time fn() for n_benchmark_iterations and return a BenchmarkRow."""
        stats = LatencyStats()
        n = self._cfg.n_benchmark_iterations
        # warmup
        for _ in range(self._cfg.n_warmup_iterations):
            try:
                fn()
            except Exception:
                pass
        # measure
        for _ in range(n):
            t0 = time.perf_counter()
            try:
                fn()
            except Exception:
                pass
            stats.add((time.perf_counter() - t0) * 1000.0)

        return BenchmarkRow(
            scenario=name,
            mean_ms=stats.mean_ms,
            p95_ms=stats.p95_ms,
            p99_ms=stats.p99_ms,
            min_ms=stats.min_ms,
            max_ms=stats.max_ms,
            iterations=stats.count,
        )

    def run(self) -> BenchmarkResult:
        rows: List[BenchmarkRow] = []

        # 1. Full pipeline
        i = [0]
        def full_pipeline():
            pred = self._make_pred(f"bench-full-{i[0]}")
            ctx = self._make_engine_ctx(f"bench-full-{i[0]}")
            i[0] += 1
            self._integrator.process_completed_prediction(ctx, pred)

        rows.append(self._time_scenario("full_pipeline", full_pipeline))

        # 2. ConfidenceThresholdStrategy
        from medical_ai_engine.anomaly.strategies import ConfidenceThresholdStrategy, StrategyConfiguration
        conf_strat = ConfidenceThresholdStrategy(StrategyConfiguration())
        conf_strat.initialize({})
        j = [0]
        def conf_bench():
            ctx = self._make_ctx(f"bench-conf-{j[0]}")
            j[0] += 1
            conf_strat.detect(ctx)
        rows.append(self._time_scenario("confidence_strategy", conf_bench))

        # 3. MahalanobisOODStrategy
        from medical_ai_engine.anomaly.strategies import MahalanobisOODStrategy
        mah_strat = MahalanobisOODStrategy(StrategyConfiguration())
        mah_strat.initialize({})
        k = [0]
        def mah_bench():
            ctx = self._make_ctx(f"bench-mah-{k[0]}")
            k[0] += 1
            mah_strat.detect(ctx)
        rows.append(self._time_scenario("mahalanobis_strategy", mah_bench))

        # 4. UnknownDiseaseDetector
        from medical_ai_engine.anomaly.unknown import UnknownDiseaseDetector
        unk_det = UnknownDiseaseDetector()
        m = [0]
        def unk_bench():
            ctx = self._make_ctx(f"bench-unk-{m[0]}")
            m[0] += 1
            unk_det.analyze(ctx)
        rows.append(self._time_scenario("unknown_detector", unk_bench))

        # 5. DecisionEngine
        from medical_ai_engine.anomaly.decision import DecisionEngine
        dec_eng = DecisionEngine()
        n_idx = [0]
        def dec_bench():
            ctx = self._make_ctx(f"bench-dec-{n_idx[0]}")
            n_idx[0] += 1
            dec_eng.evaluate_signals(prediction_result=ctx)
        rows.append(self._time_scenario("decision_fusion", dec_bench))

        # 6. Per-modality full pipeline
        for modality in self._cfg.supported_modalities:
            mod_i = [0]
            def mod_fn(mod=modality):
                pred = self._make_pred(f"bench-{mod}-{mod_i[0]}", modality=mod)
                ctx = self._make_engine_ctx(f"bench-{mod}-{mod_i[0]}")
                mod_i[0] += 1
                self._integrator.process_completed_prediction(ctx, pred)
            rows.append(self._time_scenario(f"modality_{modality}", mod_fn))

        logger.info(
            "[BenchmarkRunner] Completed %d benchmark scenarios.", len(rows)
        )
        return BenchmarkResult(rows=rows)


__all__ = ["BenchmarkRunner", "BenchmarkResult", "BenchmarkRow"]
