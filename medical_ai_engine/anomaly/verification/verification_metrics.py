"""
verification_metrics.py — Module 16.7: Anomaly Detection Verification & Benchmarking
Medical AI Platform — Latency, throughput, memory, and quality metric containers.
"""
from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class LatencyStats:
    """Statistical summary of a latency sample series."""

    samples: List[float] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.samples)

    @property
    def mean_ms(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0.0

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples) if self.samples else 0.0

    @property
    def std_ms(self) -> float:
        return statistics.stdev(self.samples) if len(self.samples) > 1 else 0.0

    @property
    def min_ms(self) -> float:
        return min(self.samples) if self.samples else 0.0

    @property
    def max_ms(self) -> float:
        return max(self.samples) if self.samples else 0.0

    @property
    def p95_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_s = sorted(self.samples)
        idx = max(0, int(len(sorted_s) * 0.95) - 1)
        return sorted_s[idx]

    @property
    def p99_ms(self) -> float:
        if not self.samples:
            return 0.0
        sorted_s = sorted(self.samples)
        idx = max(0, int(len(sorted_s) * 0.99) - 1)
        return sorted_s[idx]

    def add(self, value_ms: float) -> None:
        self.samples.append(value_ms)

    def to_dict(self) -> Dict:
        return {
            "count": self.count,
            "mean_ms": round(self.mean_ms, 3),
            "median_ms": round(self.median_ms, 3),
            "std_ms": round(self.std_ms, 3),
            "min_ms": round(self.min_ms, 3),
            "max_ms": round(self.max_ms, 3),
            "p95_ms": round(self.p95_ms, 3),
            "p99_ms": round(self.p99_ms, 3),
        }


@dataclass
class ThroughputStats:
    """Throughput measurement result."""

    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    elapsed_seconds: float = 0.0

    @property
    def rps(self) -> float:
        """Requests per second."""
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.total_requests / self.elapsed_seconds

    @property
    def success_rate(self) -> float:
        if self.total_requests == 0:
            return 0.0
        return self.successful_requests / self.total_requests

    def to_dict(self) -> Dict:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "rps": round(self.rps, 2),
            "success_rate": round(self.success_rate, 4),
        }


@dataclass
class MemoryStats:
    """RSS memory usage snapshot series."""

    baseline_mb: float = 0.0
    peak_mb: float = 0.0
    final_mb: float = 0.0
    growth_mb: float = 0.0
    snapshots: List[float] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "baseline_mb": round(self.baseline_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "final_mb": round(self.final_mb, 2),
            "growth_mb": round(self.growth_mb, 2),
        }


@dataclass
class StressTestResult:
    """Result of a single stress test tier."""

    thread_count: int = 0
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    elapsed_seconds: float = 0.0
    pass_rate: float = 0.0
    rps: float = 0.0
    mean_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    deadlock_detected: bool = False
    race_condition_detected: bool = False

    @property
    def passed(self) -> bool:
        return (
            self.pass_rate >= 0.99
            and not self.deadlock_detected
            and not self.race_condition_detected
        )

    def to_dict(self) -> Dict:
        return {
            "thread_count": self.thread_count,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
            "pass_rate": round(self.pass_rate, 4),
            "rps": round(self.rps, 2),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "deadlock_detected": self.deadlock_detected,
            "race_condition_detected": self.race_condition_detected,
            "passed": self.passed,
        }


@dataclass
class VerificationMetrics:
    """
    Aggregate metrics collector for Module 16.7 verification suite.
    Thread-safe for concurrent benchmark and stress-test collection.
    """

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # Latency breakdowns
    end_to_end_latency: LatencyStats = field(default_factory=LatencyStats)
    strategy_latency: LatencyStats = field(default_factory=LatencyStats)
    decision_fusion_latency: LatencyStats = field(default_factory=LatencyStats)
    unknown_detection_latency: LatencyStats = field(default_factory=LatencyStats)

    # Throughput
    throughput: Optional[ThroughputStats] = None

    # Memory
    memory: Optional[MemoryStats] = None

    # Stress tests
    stress_results: List[StressTestResult] = field(default_factory=list)

    # Validation flags
    architecture_compliant: bool = False
    thread_safety_verified: bool = False
    failure_recovery_verified: bool = False

    def record_latency(self, end_to_end_ms: float, strategy_ms: float = 0.0,
                       fusion_ms: float = 0.0, unknown_ms: float = 0.0) -> None:
        with self._lock:
            self.end_to_end_latency.add(end_to_end_ms)
            if strategy_ms > 0:
                self.strategy_latency.add(strategy_ms)
            if fusion_ms > 0:
                self.decision_fusion_latency.add(fusion_ms)
            if unknown_ms > 0:
                self.unknown_detection_latency.add(unknown_ms)

    def add_stress_result(self, result: StressTestResult) -> None:
        with self._lock:
            self.stress_results.append(result)

    def to_dict(self) -> Dict:
        return {
            "end_to_end_latency": self.end_to_end_latency.to_dict(),
            "strategy_latency": self.strategy_latency.to_dict(),
            "decision_fusion_latency": self.decision_fusion_latency.to_dict(),
            "unknown_detection_latency": self.unknown_detection_latency.to_dict(),
            "throughput": self.throughput.to_dict() if self.throughput else {},
            "memory": self.memory.to_dict() if self.memory else {},
            "stress_results": [r.to_dict() for r in self.stress_results],
            "architecture_compliant": self.architecture_compliant,
            "thread_safety_verified": self.thread_safety_verified,
            "failure_recovery_verified": self.failure_recovery_verified,
        }


__all__ = [
    "LatencyStats",
    "ThroughputStats",
    "MemoryStats",
    "StressTestResult",
    "VerificationMetrics",
]
