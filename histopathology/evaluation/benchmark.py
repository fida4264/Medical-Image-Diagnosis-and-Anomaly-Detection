"""
histopathology/evaluation/benchmark.py
=========================================
Latency benchmark for histopathology model inference.
Mirrors Module 7/8/9 benchmark.py pattern.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def run_latency_benchmark(
    model: nn.Module,
    device: torch.device,
    output_path: Path,
    iterations: int = 50,
    input_size: tuple = (1, 3, 224, 224),
) -> dict:
    """
    Measure single-image inference latency for the histopathology model.
    Saves benchmark results to a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    dummy = torch.randn(*input_size).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(dummy)

    # Benchmark
    latencies_ms = []
    with torch.no_grad():
        for _ in range(iterations):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies_ms.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(latencies_ms) / len(latencies_ms)
    p50 = sorted(latencies_ms)[len(latencies_ms) // 2]
    p95 = sorted(latencies_ms)[int(len(latencies_ms) * 0.95)]

    results = {
        "module": "histopathology",
        "device": str(device),
        "iterations": iterations,
        "mean_latency_ms": round(mean_ms, 3),
        "p50_latency_ms": round(p50, 3),
        "p95_latency_ms": round(p95, 3),
        "min_latency_ms": round(min(latencies_ms), 3),
        "max_latency_ms": round(max(latencies_ms), 3),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(
        f"Benchmark complete: mean={mean_ms:.2f}ms, p50={p50:.2f}ms, p95={p95:.2f}ms"
    )
    return results
