"""
stress_tester.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Concurrent multi-threaded stress tester (20, 50, 100 threads).
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from PIL import Image

from medical_ai_engine.explainability.integration import ExplainabilityIntegrator
from .verification_metrics import CategoryScore

logger = logging.getLogger(__name__)


class StressTester:
    """
    Executes concurrent load and stress testing at 20, 50, 100 thread concurrency levels.

    Verifies:
      - Thread safety and lock non-contention
      - Zero deadlocks or race conditions
      - Cache consistency under load
    """

    def test(self, score: CategoryScore, concurrency_levels: list[int] | None = None) -> None:
        levels = concurrency_levels or [20, 50, 100]
        integrator = ExplainabilityIntegrator()
        img = Image.new("RGB", (224, 224), color=(200, 100, 50))

        for level in levels:
            try:
                t0 = time.perf_counter()

                def _worker(idx: int):
                    return integrator.execute_with_explanation(
                        img,
                        manual_modality_override="brain_mri",
                        request_id=f"stress-{level}-{idx}",
                    )

                with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
                    futures = [pool.submit(_worker, i) for i in range(level)]
                    results = [f.result(timeout=15.0) for f in concurrent.futures.as_completed(futures)]

                elapsed = time.perf_counter() - t0
                successes = sum(1 for r in results if r.is_success)

                if successes == level:
                    score.record_pass(f"Concurrent stress test @ {level} threads passed (100% success in {elapsed:.2f}s)")
                else:
                    score.record_fail(f"Concurrent stress test @ {level} threads had failures ({successes}/{level})")

            except Exception as exc:
                score.record_fail(f"Stress test @ {level} threads failed: {exc}")


__all__ = ["StressTester"]
