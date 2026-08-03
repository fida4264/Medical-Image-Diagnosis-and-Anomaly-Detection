"""
brain_ct/evaluation/benchmark.py
================================
Inference latency and throughput benchmarking for Brain CT.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def benchmark_inference(
    model: nn.Module,
    input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    num_runs: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> Dict[str, float]:
    """Measures p50/p95/p99 latency and FPS throughput."""
    dev = torch.device(device)
    model.to(dev).eval()
    dummy = torch.randn(*input_size, device=dev)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy)

    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy)
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    arr = np.array(latencies_ms)
    mean_ms = float(np.mean(arr))
    results = {
        "mean_latency_ms": round(mean_ms, 2),
        "p50_latency_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_latency_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_latency_ms": round(float(np.percentile(arr, 99)), 2),
        "throughput_fps": round(1000.0 / (mean_ms + 1e-8), 1),
        "num_runs": num_runs,
        "device": device,
    }
    logger.info(f"Benchmark: p50={results['p50_latency_ms']}ms, FPS={results['throughput_fps']}")
    return results
