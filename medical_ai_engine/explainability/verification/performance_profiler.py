"""
performance_profiler.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Profiles per-stage latency breakdown for explainability workflow.
"""

from __future__ import annotations

import logging
import time
from PIL import Image

from medical_ai_engine.explainability.integration import ExplainabilityIntegrator
from .verification_metrics import CategoryScore, VerificationMetrics

logger = logging.getLogger(__name__)


class PerformanceProfiler:
    """
    Profiles end-to-end and per-stage latency for the Explainability Framework.
    """

    def profile(self, score: CategoryScore, metrics: VerificationMetrics) -> None:
        try:
            integrator = ExplainabilityIntegrator()
            dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))

            # Warmup run
            integrator.execute_with_explanation(dummy_img, manual_modality_override="brain_mri")

            # 10 benchmark iterations
            latencies = []
            for i in range(10):
                t0 = time.perf_counter()
                res = integrator.execute_with_explanation(dummy_img, manual_modality_override="brain_mri")
                elapsed_ms = (time.perf_counter() - t0) * 1000.0
                latencies.append(elapsed_ms)
                metrics.record_latency(elapsed_ms)

            avg_ms = sum(latencies) / len(latencies)
            score.record_pass(f"Latency benchmark complete (10 runs, mean: {avg_ms:.2f} ms)")

            if avg_ms < 100.0:
                score.record_pass(f"Latency meets target (<100 ms threshold: {avg_ms:.2f} ms)")
            else:
                score.record_fail(f"Latency exceeds target threshold (got {avg_ms:.2f} ms)")

        except Exception as exc:
            score.record_fail(f"PerformanceProfiler exception: {exc}")


__all__ = ["PerformanceProfiler"]
