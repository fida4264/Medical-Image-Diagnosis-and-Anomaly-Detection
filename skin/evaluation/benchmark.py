"""
skin/evaluation/benchmark.py
=============================
Inference throughput and latency benchmarking engine for Module 12.
"""

from __future__ import annotations

import logging
import time
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BenchmarkEngine:
    """Measures single-batch inference latency and FPS throughput."""

    def __init__(self, model: nn.Module, iterations: int = 50, batch_size: int = 1):
        self.model = model
        self.iterations = iterations
        self.batch_size = batch_size
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def run(self) -> Dict[str, float]:
        """Run benchmark loop and return latency statistics."""
        self.model.eval()
        dummy_input = torch.randn(self.batch_size, 3, 224, 224).to(self.device)

        # Warmup
        for _ in range(10):
            _ = self.model(dummy_input)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(self.iterations):
            t0 = time.perf_counter()
            _ = self.model(dummy_input)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

        mean_ms = float(sum(times) / len(times))
        fps = float(1000.0 / mean_ms) if mean_ms > 0 else 0.0

        results = {
            "mean_latency_ms": round(mean_ms, 3),
            "fps_throughput": round(fps, 2),
            "iterations": self.iterations,
            "device": self.device.type,
        }
        logger.info(f"Benchmark: Latency = {mean_ms:.2f} ms | Throughput = {fps:.1f} FPS")
        return results
