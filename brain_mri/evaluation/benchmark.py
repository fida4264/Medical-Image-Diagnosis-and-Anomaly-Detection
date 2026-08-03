"""
brain_mri/evaluation/benchmark.py
==================================
Inference latency and throughput benchmarking for Brain MRI Model.
Calculates p50, p95, p99 latency percentiles and FPS throughput.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Any, Tuple

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
    """
    Measures inference throughput (FPS) and latency percentiles.
    """
    dev = torch.device(device)
    model.to(dev)
    model.eval()

    dummy_input = torch.randn(*input_size, device=dev)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(dummy_input)

    # Benchmark run
    latencies_ms = []
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = model(dummy_input)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    latencies_arr = np.array(latencies_ms)
    p50 = float(np.percentile(latencies_arr, 50))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_latency = float(np.mean(latencies_arr))
    throughput_fps = float(1000.0 / (mean_latency + 1e-8))

    results = {
        "mean_latency_ms": round(mean_latency, 2),
        "p50_latency_ms": round(p50, 2),
        "p95_latency_ms": round(p95, 2),
        "p99_latency_ms": round(p99, 2),
        "throughput_fps": round(throughput_fps, 1),
        "num_runs": num_runs,
        "device": device,
    }

    logger.info(
        f"Benchmark on {device}: Mean={p50}ms, p95={p95}ms, Throughput={throughput_fps} FPS"
    )
    return results
