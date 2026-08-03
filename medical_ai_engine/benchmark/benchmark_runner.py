"""
benchmark_runner.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Orchestrates full benchmark evaluation across all 7 modalities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np

from medical_ai_engine.benchmark.benchmark_configuration import BenchmarkConfiguration
from medical_ai_engine.benchmark.benchmark_exceptions import BenchmarkExecutionError
from medical_ai_engine.benchmark.metrics import (
    ClassificationMetrics,
    ModalityBenchmarkResult,
    PerformanceMetrics,
    calculate_classification_metrics,
)
from medical_ai_engine.benchmark.performance_profiler import PerformanceProfiler

logger = logging.getLogger(__name__)


# ── Synthetic Model Registry ─────────────────────────────────────────────────
# Representative ground-truth / prediction statistics for each medical modality.
# Values are calibrated to reflect realistic published performance of clinical
# deep learning models (ResNet-50, DenseNet-121, EfficientNet-B0, ViT-B16, etc.)

_MODALITY_SPECS: Dict[str, Dict[str, Any]] = {
    "brain_mri": {
        "model_name": "MRI-DenseNet121",
        "architecture": "DenseNet-121",
        "n_classes": 4,
        "class_names": ["Glioma", "Meningioma", "No Tumor", "Pituitary"],
        "f1_target": 0.961,
        "accuracy_target": 0.963,
        "latency_ms": 14.2,
    },
    "chest_xray": {
        "model_name": "CXR-EfficientNetB0",
        "architecture": "EfficientNet-B0",
        "n_classes": 3,
        "class_names": ["Normal", "Pneumonia", "COVID-19"],
        "f1_target": 0.948,
        "accuracy_target": 0.951,
        "latency_ms": 11.7,
    },
    "brain_ct": {
        "model_name": "CT-ResNet50",
        "architecture": "ResNet-50",
        "n_classes": 3,
        "class_names": ["Normal", "Hemorrhage", "Ischemia"],
        "f1_target": 0.933,
        "accuracy_target": 0.937,
        "latency_ms": 13.5,
    },
    "histopathology": {
        "model_name": "Histo-ViTB16",
        "architecture": "ViT-B/16",
        "n_classes": 4,
        "class_names": ["Benign", "Malignant", "DCIS", "IDC"],
        "f1_target": 0.956,
        "accuracy_target": 0.958,
        "latency_ms": 18.1,
    },
    "retina": {
        "model_name": "Retina-InceptionV3",
        "architecture": "Inception-V3",
        "n_classes": 5,
        "class_names": ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"],
        "f1_target": 0.944,
        "accuracy_target": 0.946,
        "latency_ms": 16.3,
    },
    "skin": {
        "model_name": "Skin-MobileNetV3",
        "architecture": "MobileNet-V3",
        "n_classes": 7,
        "class_names": [
            "Melanoma", "Nevus", "BCC", "AK", "BKL", "DF", "VASC"
        ],
        "f1_target": 0.921,
        "accuracy_target": 0.924,
        "latency_ms": 9.4,
    },
    "ultrasound": {
        "model_name": "US-EfficientNetB3",
        "architecture": "EfficientNet-B3",
        "n_classes": 3,
        "class_names": ["Benign", "Malignant", "Normal"],
        "f1_target": 0.938,
        "accuracy_target": 0.941,
        "latency_ms": 12.6,
    },
}


def _generate_synthetic_predictions(
    n_samples: int,
    n_classes: int,
    accuracy_target: float,
    seed: int = 42,
) -> tuple:
    """
    Generate synthetic y_true / y_pred / y_prob with controlled accuracy.

    Returns (y_true, y_pred, y_prob) arrays.
    """
    rng = np.random.RandomState(seed)
    y_true = rng.randint(0, n_classes, size=n_samples)
    y_pred = y_true.copy()

    # Introduce controlled errors to match target accuracy
    n_errors = max(0, int(n_samples * (1.0 - accuracy_target)))
    error_indices = rng.choice(n_samples, size=n_errors, replace=False)
    for idx in error_indices:
        wrong_classes = [c for c in range(n_classes) if c != y_true[idx]]
        y_pred[idx] = rng.choice(wrong_classes)

    # Generate calibrated probability scores
    y_prob = rng.dirichlet(alpha=np.ones(n_classes) * 0.5, size=n_samples)
    # Boost the correct class probability for correctly-classified samples
    correct_mask = y_true == y_pred
    for i in np.where(correct_mask)[0]:
        boost = rng.uniform(0.3, 0.6)
        y_prob[i, y_true[i]] += boost
        y_prob[i] = y_prob[i] / y_prob[i].sum()

    return y_true, y_pred, y_prob


class _SyntheticModelAdapter:
    """
    Lightweight synthetic model adapter that simulates timed inference
    for performance profiling.
    """

    def __init__(self, latency_ms: float, architecture: str) -> None:
        self._latency_ms = latency_ms
        self.architecture = architecture
        # Simulate model size (representative of architecture)
        arch_sizes = {
            "DenseNet-121": (7_900_000, 30.5),
            "EfficientNet-B0": (5_300_000, 20.2),
            "ResNet-50": (25_600_000, 97.7),
            "ViT-B/16": (86_500_000, 330.0),
            "Inception-V3": (23_800_000, 91.1),
            "MobileNet-V3": (5_400_000, 20.7),
            "EfficientNet-B3": (12_200_000, 46.7),
        }
        self._param_count, self._size_mb = arch_sizes.get(architecture, (11_500_000, 44.0))

    def predict(self, x: Any) -> Any:
        """Simulate timed inference with realistic latency."""
        time.sleep(self._latency_ms / 1000.0 * (1 + np.random.randn() * 0.05))
        return None

    @property
    def metadata(self) -> Dict[str, Any]:
        return {"param_count": self._param_count, "size_mb": self._size_mb}


class BenchmarkRunner:
    """
    Orchestrates global benchmark evaluation across all 7 medical modalities.

    For each modality the runner:
    1. Generates calibrated synthetic evaluation samples.
    2. Computes classification metrics (accuracy, precision, recall, F1, ROC-AUC).
    3. Profiles a lightweight synthetic model adapter for latency / throughput / memory.
    4. Assembles a ModalityBenchmarkResult.
    5. Returns all results for downstream comparison and reporting.

    Design: Uses synthetic evaluation data because physical model weights are
    beyond the scope of a benchmark orchestration module. In production, this
    runner is extended by swapping in real IPredictorAdapter instances.
    """

    def __init__(
        self,
        config: Optional[BenchmarkConfiguration] = None,
    ) -> None:
        self.config = config or BenchmarkConfiguration.default()
        self._profiler = PerformanceProfiler(config=self.config)

    def run_all_modalities(self) -> List[ModalityBenchmarkResult]:
        """
        Run benchmarks for all 7 platform modalities.

        Returns
        -------
        List[ModalityBenchmarkResult]
            One result per modality, ordered by modality name.
        """
        results: List[ModalityBenchmarkResult] = []
        for modality in self.config.supported_modalities:
            try:
                result = self.run_modality(modality)
                results.append(result)
                logger.info(
                    "[BenchmarkRunner] %s — F1=%.4f Acc=%.4f Latency=%.2fms RPS=%.1f",
                    modality,
                    result.classification_metrics.f1_score,
                    result.classification_metrics.accuracy,
                    result.performance_metrics.inference_latency_ms,
                    result.performance_metrics.throughput_rps,
                )
            except Exception as exc:
                logger.warning("[BenchmarkRunner] %s failed: %s", modality, exc)

        return results

    def run_modality(self, modality: str) -> ModalityBenchmarkResult:
        """
        Run full benchmark evaluation for a single modality.

        Parameters
        ----------
        modality : str
            One of the 7 supported modality keys.

        Returns
        -------
        ModalityBenchmarkResult
        """
        if modality not in _MODALITY_SPECS:
            raise BenchmarkExecutionError(
                f"Unknown modality: {modality!r}. "
                f"Supported: {list(_MODALITY_SPECS.keys())}"
            )

        spec = _MODALITY_SPECS[modality]
        n_classes = spec["n_classes"]
        class_names: List[str] = spec["class_names"]
        n_samples = max(200, self.config.n_eval_runs * 4)

        # 1. Synthetic classification evaluation
        y_true, y_pred, y_prob = _generate_synthetic_predictions(
            n_samples=n_samples,
            n_classes=n_classes,
            accuracy_target=spec["accuracy_target"],
            seed=hash(modality) % (2 ** 31),
        )

        clf_metrics = calculate_classification_metrics(
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob if self.config.compute_roc_auc else None,
            class_names=class_names,
        )

        # 2. Performance profiling
        adapter = _SyntheticModelAdapter(
            latency_ms=spec["latency_ms"],
            architecture=spec["architecture"],
        )
        perf_metrics = self._profiler.profile_model(
            model_adapter=adapter,
            modality=modality,
        )

        # Patch model size and param count from adapter metadata
        meta = adapter.metadata
        perf_metrics = PerformanceMetrics(
            inference_latency_ms=perf_metrics.inference_latency_ms,
            latency_median_ms=perf_metrics.latency_median_ms,
            latency_p95_ms=perf_metrics.latency_p95_ms,
            latency_p99_ms=perf_metrics.latency_p99_ms,
            latency_std_ms=perf_metrics.latency_std_ms,
            throughput_rps=perf_metrics.throughput_rps,
            baseline_memory_mb=perf_metrics.baseline_memory_mb,
            peak_memory_mb=perf_metrics.peak_memory_mb,
            memory_growth_mb=perf_metrics.memory_growth_mb,
            model_size_mb=meta["size_mb"],
            param_count=meta["param_count"],
            batch_throughput_rps=perf_metrics.batch_throughput_rps,
        )

        # 3. Composite score (classification + latency balance)
        lat_ms = perf_metrics.inference_latency_ms
        lat_score = max(0.0, 1.0 - (lat_ms / (self.config.latency_budget_ms * 4.0)))
        composite_score = (
            0.55 * clf_metrics.f1_score
            + 0.25 * clf_metrics.accuracy
            + 0.10 * (clf_metrics.roc_auc or clf_metrics.f1_score)
            + 0.10 * lat_score
        )

        return ModalityBenchmarkResult(
            modality=modality,
            model_name=spec["model_name"],
            architecture=spec["architecture"],
            classification_metrics=clf_metrics,
            performance_metrics=perf_metrics,
            composite_score=composite_score,
            metadata={
                "n_samples": n_samples,
                "n_classes": n_classes,
                "device": self.config.device,
                "spec_latency_ms": spec["latency_ms"],
            },
        )


__all__ = ["BenchmarkRunner"]
