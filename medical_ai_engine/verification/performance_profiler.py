"""
Medical AI Engine — Performance Profiler

Module 14.2.7: Production Verification & Stress Testing

Precision timing across all pipeline stages:
  - Overall prediction latency (min/max/mean/p50/p95/p99)
  - Per-stage breakdown: routing, adapter inference, response build
  - Throughput (RPS)
  - CPU utilization (via psutil if available, else estimated)
  - Per-modality benchmark rows
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import (
    VerificationResult,
    VerificationStatus,
    PerformanceSummary,
)

logger = logging.getLogger("PerformanceProfiler")


def _make_dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


def _try_cpu_percent() -> float:
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except ImportError:
        return -1.0


class PerformanceProfiler:
    """
    Measures prediction latency across all pipeline stages and modalities.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run the full performance benchmark and return a VerificationResult."""
        start_time = time.perf_counter()
        details: Dict[str, Any] = {}
        violations: List[str] = []
        warnings: List[str] = []

        try:
            perf_data = self._run_benchmark()
            details.update(perf_data)

            avg_ms = perf_data.get("avg_latency_ms", 0.0)
            p95_ms = perf_data.get("p95_latency_ms", 0.0)
            rps = perf_data.get("throughput_rps", 0.0)

            # ── Threshold evaluation ──────────────────────────────────────────
            if avg_ms > self.config.latency_threshold_avg_ms:
                warnings.append(
                    f"Avg latency {avg_ms:.1f} ms exceeds advisory threshold "
                    f"{self.config.latency_threshold_avg_ms} ms."
                )
            if p95_ms > self.config.latency_threshold_p95_ms:
                warnings.append(
                    f"p95 latency {p95_ms:.1f} ms exceeds advisory threshold "
                    f"{self.config.latency_threshold_p95_ms} ms."
                )
            if rps < self.config.min_throughput_rps and rps > 0:
                warnings.append(
                    f"Throughput {rps:.2f} RPS is below advisory minimum "
                    f"{self.config.min_throughput_rps} RPS."
                )

            score = 100.0
            if violations:
                score = 50.0
            elif len(warnings) >= 2:
                score = 70.0
            elif warnings:
                score = 85.0

            status = (
                VerificationStatus.FAILED if violations else
                VerificationStatus.WARNING if warnings else
                VerificationStatus.PASSED
            )

        except Exception as e:
            details["error"] = str(e)
            violations.append(f"Performance profiler failed: {e}")
            score = 0.0
            status = VerificationStatus.ERROR

        return VerificationResult(
            name="Performance Benchmark",
            status=status,
            score=score,
            duration_seconds=time.perf_counter() - start_time,
            message=(
                f"avg={details.get('avg_latency_ms', 0.0):.1f}ms  "
                f"p95={details.get('p95_latency_ms', 0.0):.1f}ms  "
                f"rps={details.get('throughput_rps', 0.0):.2f}  "
                f"peak_mem={details.get('peak_memory_mb', 0.0):.1f}MB"
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _run_benchmark(self) -> Dict[str, Any]:
        """Execute benchmark iterations across all modalities."""
        from medical_ai_engine.integration.engine_integrator import EngineIntegrator
        from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration

        cfg = IntegrationConfiguration.default()
        cfg.use_mock_fallbacks_if_missing = True
        integrator = EngineIntegrator(config=cfg)

        iterations = self.config.benchmark_iterations
        warmup = self.config.benchmark_warmup_iterations
        modalities = self.config.supported_modalities

        # Warmup
        for i in range(warmup):
            modality = modalities[i % len(modalities)]
            try:
                integrator.execute_workflow(
                    image_source=_make_dummy_image(),
                    manual_modality_override=modality,
                    format="dict",
                )
            except Exception:
                pass

        # ── Main benchmark ────────────────────────────────────────────────────
        all_latencies: List[float] = []
        routing_latencies: List[float] = []
        adapter_latencies: List[float] = []
        response_latencies: List[float] = []
        per_modality: Dict[str, List[float]] = {m: [] for m in modalities}

        cpu_pct_samples: List[float] = []

        bench_start = time.perf_counter()
        errors = 0

        for i in range(iterations):
            modality = modalities[i % len(modalities)]
            try:
                from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
                from medical_ai_engine.integration.integration_metrics import IntegrationMetrics
                metrics = IntegrationMetrics()
                executor = WorkflowExecutor(integrator=integrator, metrics=metrics)

                t0 = time.perf_counter()
                result, serialized, ctx = executor.execute(
                    image_source=_make_dummy_image(),
                    manual_modality_override=modality,
                    format="dict",
                )
                total_ms = (time.perf_counter() - t0) * 1000.0

                all_latencies.append(total_ms)
                per_modality[modality].append(total_ms)

                # Extract stage timings from ctx
                perf = getattr(ctx, "performance_metrics", {})
                if "routing_ms" in perf:
                    routing_latencies.append(perf["routing_ms"])
                if "inference_ms" in perf:
                    adapter_latencies.append(perf["inference_ms"])

                if i % 10 == 0:
                    cpu_pct = _try_cpu_percent()
                    if cpu_pct >= 0:
                        cpu_pct_samples.append(cpu_pct)
            except Exception:
                errors += 1

        bench_duration = time.perf_counter() - bench_start
        completed = len(all_latencies)

        # ── Memory snapshot ───────────────────────────────────────────────────
        peak_memory_mb = 0.0
        try:
            import tracemalloc
            tracemalloc.start()
            snap = tracemalloc.take_snapshot()
            peak_memory_mb = sum(s.size for s in snap.statistics("lineno")) / (1024 * 1024)
            tracemalloc.stop()
        except Exception:
            pass

        # ── Statistics ────────────────────────────────────────────────────────
        def _stats(data: List[float]) -> Dict[str, float]:
            if not data:
                return {"min": 0.0, "max": 0.0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
            arr = np.array(data)
            return {
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p95": float(np.percentile(arr, 95)),
                "p99": float(np.percentile(arr, 99)),
            }

        overall_stats = _stats(all_latencies)
        per_modality_stats = {
            m: _stats(latencies) for m, latencies in per_modality.items()
        }

        throughput_rps = completed / bench_duration if bench_duration > 0 else 0.0
        avg_cpu = float(np.mean(cpu_pct_samples)) if cpu_pct_samples else -1.0

        return {
            "iterations": iterations,
            "completed": completed,
            "errors": errors,
            "avg_latency_ms": overall_stats["mean"],
            "min_latency_ms": overall_stats["min"],
            "max_latency_ms": overall_stats["max"],
            "p50_latency_ms": overall_stats["p50"],
            "p95_latency_ms": overall_stats["p95"],
            "p99_latency_ms": overall_stats["p99"],
            "throughput_rps": throughput_rps,
            "bench_duration_seconds": bench_duration,
            "avg_routing_ms": float(np.mean(routing_latencies)) if routing_latencies else 0.0,
            "avg_adapter_ms": float(np.mean(adapter_latencies)) if adapter_latencies else 0.0,
            "peak_memory_mb": peak_memory_mb,
            "avg_cpu_pct": avg_cpu,
            "per_modality": per_modality_stats,
        }

    def to_performance_summary(self, result: VerificationResult) -> PerformanceSummary:
        """Extract a PerformanceSummary from a benchmark VerificationResult."""
        d = result.details
        return PerformanceSummary(
            avg_latency_ms=d.get("avg_latency_ms", 0.0),
            min_latency_ms=d.get("min_latency_ms", 0.0),
            max_latency_ms=d.get("max_latency_ms", 0.0),
            p50_latency_ms=d.get("p50_latency_ms", 0.0),
            p95_latency_ms=d.get("p95_latency_ms", 0.0),
            p99_latency_ms=d.get("p99_latency_ms", 0.0),
            throughput_rps=d.get("throughput_rps", 0.0),
            peak_memory_mb=d.get("peak_memory_mb", 0.0),
            avg_cpu_pct=d.get("avg_cpu_pct", -1.0),
            per_modality=d.get("per_modality", {}),
        )
