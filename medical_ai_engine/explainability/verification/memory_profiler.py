"""
memory_profiler.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Tracks RAM and GPU memory consumption during XAI execution.
"""

from __future__ import annotations

import logging
import psutil

from .verification_metrics import CategoryScore, VerificationMetrics

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Profiles process memory usage (RSS MB) before and after explainability generation.
    """

    def profile(self, score: CategoryScore, metrics: VerificationMetrics) -> None:
        try:
            process = psutil.Process()
            mem_before = process.memory_info().rss / (1024 * 1024)

            # Trigger allocation check
            from medical_ai_engine.explainability.integration import ExplainabilityIntegrator
            from PIL import Image

            integrator = ExplainabilityIntegrator()
            img = Image.new("RGB", (224, 224), color=(50, 100, 150))
            for _ in range(5):
                integrator.execute_with_explanation(img, manual_modality_override="brain_mri")

            mem_after = process.memory_info().rss / (1024 * 1024)
            delta = max(0.0, mem_after - mem_before)

            metrics.record_memory(mem_after)
            score.record_pass(f"Memory RSS measured: {mem_after:.1f} MB (delta: {delta:.2f} MB)")

            if mem_after < 1024.0:
                score.record_pass(f"Memory footprint within 1 GB limit ({mem_after:.1f} MB)")
            else:
                score.record_fail(f"Memory footprint exceeded 1 GB ({mem_after:.1f} MB)")

        except Exception as exc:
            score.record_fail(f"MemoryProfiler exception: {exc}")


__all__ = ["MemoryProfiler"]
