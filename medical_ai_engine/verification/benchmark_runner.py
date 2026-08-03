"""
Medical AI Engine — Benchmark Runner

Module 14.2.7: Production Verification & Stress Testing

Runs per-modality latency benchmarks and generates a structured benchmark table.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import VerificationResult, VerificationStatus

logger = logging.getLogger("BenchmarkRunner")


def _dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


class BenchmarkRunner:
    """
    Executes structured per-modality latency benchmarks.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run per-modality benchmarks and return a VerificationResult."""
        start = time.perf_counter()
        details: Dict[str, Any] = {}
        violations: List[str] = []
        warnings: List[str] = []

        try:
            benchmark_data = self._run_all_modalities()
            details.update(benchmark_data)

            # Simple pass/warn evaluation
            for modality, stats in benchmark_data.get("modality_benchmarks", {}).items():
                avg = stats.get("avg_ms", 0.0)
                if avg > self.config.latency_threshold_avg_ms:
                    warnings.append(
                        f"Modality '{modality}' avg latency {avg:.1f} ms "
                        f"(threshold {self.config.latency_threshold_avg_ms} ms)"
                    )

            score = 100.0 - (len(warnings) * 5.0)
            score = max(score, 60.0) if not violations else max(score, 0.0)
            status = (
                VerificationStatus.FAILED if violations else
                VerificationStatus.WARNING if warnings else
                VerificationStatus.PASSED
            )

        except Exception as e:
            details["error"] = str(e)
            violations.append(f"Benchmark runner failed: {e}")
            score = 0.0
            status = VerificationStatus.ERROR

        return VerificationResult(
            name="Per-Modality Benchmark",
            status=status,
            score=score,
            duration_seconds=time.perf_counter() - start,
            message=(
                f"Benchmarked {len(self.config.supported_modalities)} modalities, "
                f"{details.get('total_iterations', 0)} total iterations. "
                f"Global avg={details.get('global_avg_ms', 0.0):.1f}ms, "
                f"rps={details.get('global_rps', 0.0):.2f}"
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _run_all_modalities(self) -> Dict[str, Any]:
        from medical_ai_engine.integration.engine_integrator import EngineIntegrator
        from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration

        cfg = IntegrationConfiguration.default()
        cfg.use_mock_fallbacks_if_missing = True
        integrator = EngineIntegrator(config=cfg)

        n = self.config.benchmark_iterations
        warmup = self.config.benchmark_warmup_iterations
        modality_benchmarks: Dict[str, Dict[str, float]] = {}
        all_latencies: List[float] = []

        bench_start = time.perf_counter()

        for modality in self.config.supported_modalities:
            # Warmup
            for _ in range(warmup):
                try:
                    integrator.execute_workflow(
                        image_source=_dummy_image(),
                        manual_modality_override=modality,
                        format="dict",
                    )
                except Exception:
                    pass

            latencies: List[float] = []
            errors = 0

            for _ in range(n):
                try:
                    t0 = time.perf_counter()
                    integrator.execute_workflow(
                        image_source=_dummy_image(),
                        manual_modality_override=modality,
                        format="dict",
                    )
                    latencies.append((time.perf_counter() - t0) * 1000.0)
                except Exception:
                    errors += 1

            if latencies:
                arr = np.array(latencies)
                modality_benchmarks[modality] = {
                    "avg_ms": float(np.mean(arr)),
                    "min_ms": float(np.min(arr)),
                    "max_ms": float(np.max(arr)),
                    "p50_ms": float(np.percentile(arr, 50)),
                    "p95_ms": float(np.percentile(arr, 95)),
                    "iterations": len(latencies),
                    "errors": errors,
                }
                all_latencies.extend(latencies)
            else:
                modality_benchmarks[modality] = {
                    "avg_ms": 0.0, "min_ms": 0.0, "max_ms": 0.0,
                    "p50_ms": 0.0, "p95_ms": 0.0,
                    "iterations": 0, "errors": errors,
                }

        total_duration = time.perf_counter() - bench_start
        total_iter = len(all_latencies)
        global_avg = float(np.mean(all_latencies)) if all_latencies else 0.0
        global_rps = total_iter / total_duration if total_duration > 0 else 0.0

        return {
            "modality_benchmarks": modality_benchmarks,
            "total_iterations": total_iter,
            "global_avg_ms": global_avg,
            "global_rps": global_rps,
            "total_duration_seconds": total_duration,
            "benchmark_table": self._render_table(modality_benchmarks),
        }

    def _render_table(self, benchmarks: Dict[str, Dict[str, float]]) -> str:
        """Render a human-readable benchmark table."""
        header = f"{'Modality':<20} {'Avg(ms)':>9} {'Min(ms)':>9} {'Max(ms)':>9} {'p50(ms)':>9} {'p95(ms)':>9} {'Iters':>7}"
        sep = "-" * 78
        rows = [header, sep]
        for modality, stats in benchmarks.items():
            rows.append(
                f"{modality:<20} "
                f"{stats['avg_ms']:>9.1f} "
                f"{stats['min_ms']:>9.1f} "
                f"{stats['max_ms']:>9.1f} "
                f"{stats['p50_ms']:>9.1f} "
                f"{stats['p95_ms']:>9.1f} "
                f"{stats['iterations']:>7}"
            )
        return "\n".join(rows)
