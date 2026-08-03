"""
verification_metrics.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Performance and verification metrics collector.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class CategoryScore:
    """Score snapshot for one verification domain."""
    name: str
    passed: int = 0
    failed: int = 0
    total: int = 0
    score_pct: float = 100.0
    details: List[str] = field(default_factory=list)

    def record_pass(self, detail: str = "") -> None:
        self.passed += 1
        self.total += 1
        if detail:
            self.details.append(f"PASS: {detail}")
        self.score_pct = round((self.passed / max(self.total, 1)) * 100.0, 1)

    def record_fail(self, detail: str = "") -> None:
        self.failed += 1
        self.total += 1
        if detail:
            self.details.append(f"FAIL: {detail}")
        self.score_pct = round((self.passed / max(self.total, 1)) * 100.0, 1)


class VerificationMetrics:
    """
    Accumulates metrics across all 10 verification categories.
    """

    def __init__(self) -> None:
        self.categories: Dict[str, CategoryScore] = {
            "architecture": CategoryScore(name="Architecture Compliance"),
            "gradcam": CategoryScore(name="Grad-CAM Engine"),
            "visualization": CategoryScore(name="Heatmap & Overlay Pipeline"),
            "clinical": CategoryScore(name="Clinical Explanation Engine"),
            "benchmark": CategoryScore(name="Performance Benchmark"),
            "stress": CategoryScore(name="Stress Testing"),
            "quality": CategoryScore(name="Quality & Faithfulness"),
            "resilience": CategoryScore(name="Failure Recovery & Fault Tolerance"),
            "regression": CategoryScore(name="System Regression"),
            "integration": CategoryScore(name="End-to-End System Integration"),
        }
        self.latency_samples_ms: List[float] = []
        self.memory_samples_mb: List[float] = []

    def get_category(self, key: str) -> CategoryScore:
        return self.categories.setdefault(key, CategoryScore(name=key))

    def record_latency(self, latency_ms: float) -> None:
        self.latency_samples_ms.append(latency_ms)

    def record_memory(self, memory_mb: float) -> None:
        self.memory_samples_mb.append(memory_mb)

    def overall_score_pct(self) -> float:
        total_passed = sum(cat.passed for cat in self.categories.values())
        total_tests = sum(cat.total for cat in self.categories.values())
        if total_tests == 0:
            return 100.0
        return round((total_passed / total_tests) * 100.0, 1)

    def latency_stats(self) -> Dict[str, float]:
        if not self.latency_samples_ms:
            return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}
        samples = sorted(self.latency_samples_ms)
        p50_idx = int(len(samples) * 0.50)
        p95_idx = min(int(len(samples) * 0.95), len(samples) - 1)
        return {
            "mean_ms": round(statistics.mean(samples), 2),
            "p50_ms": round(samples[p50_idx], 2),
            "p95_ms": round(samples[p95_idx], 2),
            "max_ms": round(samples[-1], 2),
        }

    def summary(self) -> Dict[str, Any]:
        return {
            "overall_score_pct": self.overall_score_pct(),
            "categories": {k: cat.score_pct for k, cat in self.categories.items()},
            "latency": self.latency_stats(),
            "total_verifications": sum(c.total for c in self.categories.values()),
            "total_passed": sum(c.passed for c in self.categories.values()),
            "total_failed": sum(c.failed for c in self.categories.values()),
        }


__all__ = ["VerificationMetrics", "CategoryScore"]
