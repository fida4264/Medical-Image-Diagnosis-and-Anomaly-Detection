"""
Medical AI Engine — Memory Profiler

Module 14.2.7: Production Verification & Stress Testing

Runs repeated prediction cycles to detect memory leaks, uncollected cycles,
and unstable object counts.

Uses tracemalloc (stdlib) + gc to avoid external dependencies.
"""

from __future__ import annotations

import gc
import logging
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import VerificationResult, VerificationStatus

logger = logging.getLogger("MemoryProfiler")


def _make_dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


class MemoryProfiler:
    """
    Performs long-duration memory stability tests on the MedicalAIEngine platform.
    Detects memory leaks by measuring object counts and heap size before, mid, and
    after a sustained prediction cycle.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run the memory stability test and return a VerificationResult."""
        start = time.perf_counter()
        details: Dict[str, Any] = {}
        violations: List[str] = []
        warnings: List[str] = []

        try:
            result_data = self._profile_memory()
            details.update(result_data)

            # ── Evaluate results ──────────────────────────────────────────────
            obj_delta = result_data.get("object_delta", 0)
            mem_delta_mb = result_data.get("memory_delta_mb", 0.0)
            gc_unreachable = result_data.get("gc_unreachable_after", 0)

            if obj_delta > self.config.memory_leak_threshold_objects:
                warnings.append(
                    f"Object count grew by {obj_delta} (threshold: {self.config.memory_leak_threshold_objects}). "
                    "Possible memory leak."
                )
            if mem_delta_mb > self.config.memory_leak_threshold_mb:
                violations.append(
                    f"Heap grew by {mem_delta_mb:.1f} MB (threshold: {self.config.memory_leak_threshold_mb} MB). "
                    "Memory leak detected."
                )
            if gc_unreachable > 100:
                warnings.append(
                    f"GC found {gc_unreachable} unreachable objects after test. "
                    "Possible reference cycles."
                )

            score = 100.0
            if violations:
                score = 40.0
            elif warnings:
                score = 75.0

            status = (
                VerificationStatus.FAILED if violations else
                VerificationStatus.WARNING if warnings else
                VerificationStatus.PASSED
            )

        except Exception as e:
            details["error"] = str(e)
            violations.append(f"Memory profiler failed: {e}")
            score = 0.0
            status = VerificationStatus.ERROR

        return VerificationResult(
            name="Memory Stability Analysis",
            status=status,
            score=score,
            duration_seconds=time.perf_counter() - start,
            message=(
                f"Ran {details.get('cycles_completed', 0)} prediction cycles. "
                f"Object delta: {details.get('object_delta', 'N/A')}. "
                f"Heap delta: {details.get('memory_delta_mb', 0.0):.2f} MB. "
                f"{len(violations)} violation(s), {len(warnings)} warning(s)."
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _profile_memory(self) -> Dict[str, Any]:
        """Run prediction cycles with tracemalloc sampling."""
        from medical_ai_engine.integration.engine_integrator import EngineIntegrator
        from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration

        cfg = IntegrationConfiguration.default()
        cfg.use_mock_fallbacks_if_missing = True
        integrator = EngineIntegrator(config=cfg)

        max_cycles = self.config.memory_test_cycles
        max_duration = self.config.memory_test_duration_seconds

        # ── Warmup ────────────────────────────────────────────────────────────
        for _ in range(3):
            integrator.execute_workflow(
                image_source=_make_dummy_image(),
                manual_modality_override="brain_mri",
                format="dict",
            )
        gc.collect()

        # ── Baseline measurement ──────────────────────────────────────────────
        tracemalloc.start()
        gc.collect()
        obj_count_before = len(gc.get_objects())
        snapshot_before = tracemalloc.take_snapshot()

        cycle_start = time.perf_counter()
        cycles_completed = 0
        cycle_latencies: List[float] = []

        modalities = self.config.supported_modalities
        errors = 0

        # ── Test cycles ───────────────────────────────────────────────────────
        while cycles_completed < max_cycles:
            elapsed = time.perf_counter() - cycle_start
            if elapsed >= max_duration:
                break
            try:
                modality = modalities[cycles_completed % len(modalities)]
                t0 = time.perf_counter()
                integrator.execute_workflow(
                    image_source=_make_dummy_image(),
                    manual_modality_override=modality,
                    format="dict",
                )
                cycle_latencies.append((time.perf_counter() - t0) * 1000.0)
                cycles_completed += 1
            except Exception:
                errors += 1
                cycles_completed += 1

        # ── Final measurement ─────────────────────────────────────────────────
        gc_unreachable = gc.collect()
        obj_count_after = len(gc.get_objects())
        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare top memory stats
        top_stats = snapshot_after.compare_to(snapshot_before, "lineno")
        top_5 = [str(stat) for stat in top_stats[:5]]

        # Compute heap delta
        before_total = sum(stat.size for stat in snapshot_before.statistics("lineno"))
        after_total = sum(stat.size for stat in snapshot_after.statistics("lineno"))
        memory_delta_mb = (after_total - before_total) / (1024 * 1024)
        peak_memory_mb = after_total / (1024 * 1024)

        return {
            "cycles_completed": cycles_completed,
            "max_cycles": max_cycles,
            "errors": errors,
            "obj_count_before": obj_count_before,
            "obj_count_after": obj_count_after,
            "object_delta": obj_count_after - obj_count_before,
            "memory_before_mb": before_total / (1024 * 1024),
            "memory_after_mb": after_total / (1024 * 1024),
            "memory_delta_mb": memory_delta_mb,
            "peak_memory_mb": peak_memory_mb,
            "gc_unreachable_after": gc_unreachable,
            "avg_latency_ms": float(np.mean(cycle_latencies)) if cycle_latencies else 0.0,
            "top_memory_allocations": top_5,
        }
