"""
Medical AI Engine — Health Monitor

Module 14.2.7: Production Verification & Stress Testing

Generates a production health dashboard with:
  - Health Score (0–100)
  - System Status (HEALTHY / DEGRADED / CRITICAL)
  - Loaded models count
  - Cache size
  - Average latency / throughput
  - Failure rate / Success rate
  - Memory and CPU usage
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import VerificationResult, VerificationStatus

logger = logging.getLogger("HealthMonitor")


def _dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


class HealthMonitor:
    """
    Polls all integrated subsystems and computes a composite health score.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run health monitoring and return a VerificationResult."""
        start = time.perf_counter()
        details: Dict[str, Any] = {}
        violations: List[str] = []
        warnings: List[str] = []

        try:
            health_data = self._collect_health()
            details.update(health_data)

            score = health_data.get("health_score", 0.0)
            status_str = health_data.get("system_status", "UNKNOWN")

            if score < 40.0:
                violations.append(f"Health score critically low: {score:.1f}/100")
            elif score < self.config.health_score_pass_threshold:
                warnings.append(f"Health score below threshold: {score:.1f}/100 (min: {self.config.health_score_pass_threshold})")

            failure_rate = health_data.get("failure_rate", 0.0)
            if failure_rate > self.config.failure_rate_warning_threshold:
                warnings.append(f"Failure rate {failure_rate:.1%} exceeds warning threshold {self.config.failure_rate_warning_threshold:.0%}")

            status = (
                VerificationStatus.FAILED if violations else
                VerificationStatus.WARNING if warnings else
                VerificationStatus.PASSED
            )

        except Exception as e:
            details["error"] = str(e)
            violations.append(f"Health monitor failed: {e}")
            score = 0.0
            status = VerificationStatus.ERROR

        return VerificationResult(
            name="System Health Monitor",
            status=status,
            score=details.get("health_score", score if violations else 0.0),
            duration_seconds=time.perf_counter() - start,
            message=(
                f"Health Score: {details.get('health_score', 0.0):.1f}/100 "
                f"({details.get('system_status', 'UNKNOWN')}). "
                f"Models: {details.get('loaded_models', 0)}. "
                f"Success rate: {details.get('success_rate', 0.0):.1%}."
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _collect_health(self) -> Dict[str, Any]:
        """Collect health data from all subsystems."""
        from medical_ai_engine.integration.engine_integrator import EngineIntegrator
        from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
        from medical_ai_engine.integration.integration_metrics import IntegrationMetrics

        cfg = IntegrationConfiguration.default()
        cfg.use_mock_fallbacks_if_missing = True
        integrator = EngineIntegrator(config=cfg)
        metrics = IntegrationMetrics()

        # ── Run a few sample predictions to collect metrics ───────────────────
        sample_latencies: List[float] = []
        successes = 0
        failures = 0

        from medical_ai_engine.integration.workflow_executor import WorkflowExecutor
        executor = WorkflowExecutor(integrator=integrator, metrics=metrics)

        for i in range(10):
            modality = self.config.supported_modalities[i % len(self.config.supported_modalities)]
            try:
                t0 = time.perf_counter()
                result, _, _ = executor.execute(
                    image_source=_dummy_image(),
                    manual_modality_override=modality,
                    format="dict",
                )
                lat = (time.perf_counter() - t0) * 1000.0
                sample_latencies.append(lat)
                if result.is_success:
                    successes += 1
                else:
                    failures += 1
            except Exception:
                failures += 1

        total = successes + failures
        success_rate = successes / total if total > 0 else 0.0
        failure_rate = failures / total if total > 0 else 0.0
        avg_latency = float(np.mean(sample_latencies)) if sample_latencies else 0.0

        # ── Engine health check ───────────────────────────────────────────────
        engine_health = integrator.engine.get_health_status()
        engine_stats = integrator.engine.get_statistics()

        # ── Registry info ─────────────────────────────────────────────────────
        registered_modalities = integrator.registry.list_registered_modalities()
        loaded_models = len(registered_modalities)

        # ── Cache info ────────────────────────────────────────────────────────
        try:
            cache_size = len(integrator.registry.cache._cache) if hasattr(integrator.registry.cache, "_cache") else 0
        except Exception:
            cache_size = loaded_models  # fallback

        # ── CPU/Memory ────────────────────────────────────────────────────────
        memory_mb = 0.0
        cpu_pct = -1.0
        try:
            import tracemalloc
            tracemalloc.start()
            snap = tracemalloc.take_snapshot()
            memory_mb = sum(s.size for s in snap.statistics("lineno")) / (1024 * 1024)
            tracemalloc.stop()
        except Exception:
            pass
        try:
            import psutil
            cpu_pct = psutil.cpu_percent(interval=0.2)
        except ImportError:
            pass

        # ── Compute health score ──────────────────────────────────────────────
        health_score = self._compute_health_score(
            success_rate=success_rate,
            loaded_models=loaded_models,
            expected_models=len(self.config.supported_modalities),
            avg_latency_ms=avg_latency,
            engine_health=engine_health,
        )

        system_status = (
            "HEALTHY" if health_score >= 80.0 else
            "DEGRADED" if health_score >= 50.0 else
            "CRITICAL"
        )

        return {
            "health_score": health_score,
            "system_status": system_status,
            "loaded_models": loaded_models,
            "registered_modalities": registered_modalities,
            "cache_size": cache_size,
            "avg_latency_ms": avg_latency,
            "throughput_rps": (successes / max(sum(sample_latencies) / 1000.0, 0.001)) if sample_latencies else 0.0,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "memory_mb": memory_mb,
            "cpu_pct": cpu_pct,
            "engine_health": engine_health,
            "engine_stats": engine_stats,
            "metrics_snapshot": metrics.snapshot(),
        }

    def _compute_health_score(
        self,
        success_rate: float,
        loaded_models: int,
        expected_models: int,
        avg_latency_ms: float,
        engine_health: Dict[str, Any],
    ) -> float:
        """Compute a 0–100 health score from subsystem metrics."""
        score = 0.0

        # Success rate (40 pts)
        score += success_rate * 40.0

        # Model coverage (25 pts)
        coverage = min(loaded_models / expected_models, 1.0) if expected_models > 0 else 0.0
        score += coverage * 25.0

        # Engine state (20 pts)
        engine_status = engine_health.get("status", "")
        if engine_status == "healthy":
            score += 20.0
        elif engine_status == "degraded":
            score += 10.0

        # Latency (15 pts)
        if avg_latency_ms <= 0:
            score += 7.5  # Unknown — partial credit
        elif avg_latency_ms <= self.config.latency_threshold_avg_ms:
            score += 15.0
        elif avg_latency_ms <= self.config.latency_threshold_p95_ms:
            score += 7.5

        return round(min(score, 100.0), 2)
