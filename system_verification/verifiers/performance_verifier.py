"""
system_verification/verifiers/performance_verifier.py
======================================================
Measures CSV load benchmark times, DataFrame memory footprint, sample image load times,
CPU core count, and GPU availability.

Author: Medical AI System — Module 5.5
"""

import logging
import os
import time
from typing import Any, Dict, List

import cv2
import pandas as pd
from PIL import Image

from system_verification.config import (
    M1_METADATA_CSV,
    M2_MASTER_METADATA_CSV,
    PERF_CSV_LOAD_WARNING_S,
    PERF_IMAGE_LOAD_WARNING_S,
    PERF_MEMORY_WARNING_MB,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


class PerformanceVerifier:
    """Measures load times, memory consumption, CPU cores, and GPU detection."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run performance verification checks."""
        logger.info("Running PerformanceVerifier ...")
        checks: List[Dict[str, Any]] = []

        m1_load_time_s = 0.0
        m2_load_time_s = 0.0
        m2_memory_mb = 0.0
        sample_img_load_time_ms = 0.0
        cpu_cores = os.cpu_count() or 1
        gpu_available = False
        gpu_device_name = "None"

        # 1. Benchmark M1 metadata.csv load time
        if M1_METADATA_CSV.exists():
            t0 = time.perf_counter()
            df1 = pd.read_csv(M1_METADATA_CSV, low_memory=False)
            t1 = time.perf_counter()
            m1_load_time_s = round(t1 - t0, 3)

            status = "PASS" if m1_load_time_s < PERF_CSV_LOAD_WARNING_S else "WARNING"
            checks.append({
                "metric": "Module 1 CSV Load Time",
                "status": status,
                "value": f"{m1_load_time_s} s",
                "details": f"Loaded {len(df1)} rows in {m1_load_time_s}s",
            })

        # 2. Benchmark M2 master_metadata.csv load time & memory
        if M2_MASTER_METADATA_CSV.exists():
            t0 = time.perf_counter()
            df2 = pd.read_csv(M2_MASTER_METADATA_CSV, low_memory=False)
            t1 = time.perf_counter()
            m2_load_time_s = round(t1 - t0, 3)

            # Calculate memory usage in MB
            m2_memory_mb = round(df2.memory_usage(deep=True).sum() / (1024 * 1024), 2)

            status = "PASS" if m2_load_time_s < PERF_CSV_LOAD_WARNING_S else "WARNING"
            checks.append({
                "metric": "Master Metadata CSV Load Time",
                "status": status,
                "value": f"{m2_load_time_s} s",
                "details": f"Loaded {len(df2)} rows in {m2_load_time_s}s",
            })

            mem_status = "PASS" if m2_memory_mb < PERF_MEMORY_WARNING_MB else "WARNING"
            checks.append({
                "metric": "Master Metadata Memory Footprint",
                "status": mem_status,
                "value": f"{m2_memory_mb} MB",
                "details": f"DataFrame memory size: {m2_memory_mb} MB",
            })

            # Sample image load time benchmark
            if "image_path" in df2.columns and len(df2) > 0:
                sample_paths = df2["image_path"].head(10).tolist()
                valid_paths = [PROJECT_ROOT / p for p in sample_paths if (PROJECT_ROOT / p).exists()]
                if valid_paths:
                    t0_img = time.perf_counter()
                    for vp in valid_paths:
                        _ = cv2.imread(str(vp))
                    t1_img = time.perf_counter()
                    avg_s = (t1_img - t0_img) / len(valid_paths)
                    sample_img_load_time_ms = round(avg_s * 1000, 2)

                    img_status = "PASS" if avg_s < PERF_IMAGE_LOAD_WARNING_S else "WARNING"
                    checks.append({
                        "metric": "Sample Image Read Latency",
                        "status": img_status,
                        "value": f"{sample_img_load_time_ms} ms/img",
                        "details": f"Average load time over {len(valid_paths)} images: {sample_img_load_time_ms}ms",
                    })

        # 3. CPU Core Count
        checks.append({
            "metric": "CPU Core Count",
            "status": "PASS",
            "value": f"{cpu_cores} cores",
            "details": f"Available logical CPU cores: {cpu_cores}",
        })

        # 4. GPU Detection (OpenCV CUDA & PyTorch / TF if installed)
        try:
            cuda_count = cv2.cuda.getCudaEnabledDeviceCount()
            if cuda_count > 0:
                gpu_available = True
                gpu_device_name = f"OpenCV CUDA ({cuda_count} devices)"
        except Exception:
            pass

        if not gpu_available:
            try:
                import torch
                if torch.cuda.is_available():
                    gpu_available = True
                    gpu_device_name = f"PyTorch CUDA ({torch.cuda.get_device_name(0)})"
            except ImportError:
                pass

        if not gpu_available:
            try:
                import tensorflow as tf
                gpus = tf.config.list_physical_devices("GPU")
                if gpus:
                    gpu_available = True
                    gpu_device_name = f"TensorFlow GPU ({len(gpus)} devices)"
            except ImportError:
                pass

        checks.append({
            "metric": "GPU Acceleration Availability",
            "status": "PASS" if gpu_available else "WARNING",
            "value": "Available" if gpu_available else "CPU Only",
            "details": f"GPU device: {gpu_device_name}",
        })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "m1_load_time_s": m1_load_time_s,
            "m2_load_time_s": m2_load_time_s,
            "m2_memory_mb": m2_memory_mb,
            "sample_img_load_time_ms": sample_img_load_time_ms,
            "cpu_cores": cpu_cores,
            "gpu_available": gpu_available,
            "gpu_device_name": gpu_device_name,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("PerformanceVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
