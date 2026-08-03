"""
benchmark_runner.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Runs comprehensive multi-modality throughput and latency benchmarks.
"""

from __future__ import annotations

import logging
import time
from PIL import Image

from medical_ai_engine.explainability.integration import ExplainabilityIntegrator
from .verification_metrics import CategoryScore

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Executes throughput (RPS) and latency distribution benchmarks.
    """

    def run(self, score: CategoryScore) -> dict:
        results = {}
        try:
            integrator = ExplainabilityIntegrator()
            dummy_img = Image.new("RGB", (224, 224), color=(100, 100, 100))

            modalities = ["brain_mri", "chest_xray", "skin"]
            for mod in modalities:
                t0 = time.perf_counter()
                count = 5
                for _ in range(count):
                    integrator.execute_with_explanation(dummy_img, manual_modality_override=mod)
                elapsed = time.perf_counter() - t0
                rps = count / max(elapsed, 1e-5)
                avg_lat = (elapsed / count) * 1000.0

                results[mod] = {"rps": round(rps, 2), "avg_latency_ms": round(avg_lat, 2)}
                score.record_pass(f"Benchmark '{mod}': {rps:.1f} RPS, avg latency {avg_lat:.2f} ms")

        except Exception as exc:
            score.record_fail(f"BenchmarkRunner exception: {exc}")

        return results


__all__ = ["BenchmarkRunner"]
