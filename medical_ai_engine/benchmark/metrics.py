"""
metrics.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Standardized metric containers and mathematical evaluation functions.
"""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from medical_ai_engine.benchmark.benchmark_exceptions import MetricComputationError


@dataclass(frozen=True)
class ClassificationMetrics:
    """
    Diagnostic classification accuracy and clinical evaluation metrics.

    Fields
    ------
    accuracy : float
        Overall classification accuracy (0.0 to 1.0).
    precision : float
        Macro-averaged precision (0.0 to 1.0).
    recall : float
        Macro-averaged recall / sensitivity (0.0 to 1.0).
    f1_score : float
        Macro-averaged F1-score (0.0 to 1.0).
    roc_auc : Optional[float]
        ROC-AUC score (0.0 to 1.0) if probability scores available, else None.
    confusion_matrix : List[List[int]]
        Square confusion matrix [[TN, FP], [FN, TP]] or multi-class counts.
    n_samples : int
        Total number of evaluation samples processed.
    class_names : List[str]
        Target disease class names in matrix index order.
    per_class_f1 : Dict[str, float]
        F1-score breakdown per individual disease class.
    """

    accuracy: float
    precision: float
    recall: float
    f1_score: float
    roc_auc: Optional[float] = None
    confusion_matrix: List[List[int]] = field(default_factory=list)
    n_samples: int = 0
    class_names: List[str] = field(default_factory=list)
    per_class_f1: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to JSON-serializable dictionary."""
        d = asdict(self)
        if self.roc_auc is not None:
            d["roc_auc"] = round(self.roc_auc, 4)
        d["accuracy"] = round(self.accuracy, 4)
        d["precision"] = round(self.precision, 4)
        d["recall"] = round(self.recall, 4)
        d["f1_score"] = round(self.f1_score, 4)
        d["per_class_f1"] = {k: round(v, 4) for k, v in self.per_class_f1.items()}
        return d


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Computational performance and efficiency metrics.

    Fields
    ------
    inference_latency_ms : float
        Mean inference latency per single image (ms).
    latency_median_ms : float
        Median inference latency (ms).
    latency_p95_ms : float
        95th percentile latency (ms).
    latency_p99_ms : float
        99th percentile latency (ms).
    latency_std_ms : float
        Standard deviation of latency (ms).
    throughput_rps : float
        Sustained throughput in Requests-Per-Second (RPS).
    baseline_memory_mb : float
        Baseline RSS memory usage (MB).
    peak_memory_mb : float
        Peak RSS memory usage during inference (MB).
    memory_growth_mb : float
        Net memory growth (MB).
    model_size_mb : float
        Model weight artifact size on disk (MB).
    param_count : int
        Total trainable and non-trainable parameter count.
    batch_throughput_rps : Dict[int, float]
        Throughput scaling across batch sizes {batch_size: RPS}.
    """

    inference_latency_ms: float
    latency_median_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    latency_std_ms: float
    throughput_rps: float
    baseline_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    model_size_mb: float = 0.0
    param_count: int = 0
    batch_throughput_rps: Dict[int, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metrics to JSON-serializable dictionary."""
        d = asdict(self)
        d["inference_latency_ms"] = round(self.inference_latency_ms, 3)
        d["latency_median_ms"] = round(self.latency_median_ms, 3)
        d["latency_p95_ms"] = round(self.latency_p95_ms, 3)
        d["latency_p99_ms"] = round(self.latency_p99_ms, 3)
        d["latency_std_ms"] = round(self.latency_std_ms, 3)
        d["throughput_rps"] = round(self.throughput_rps, 2)
        d["baseline_memory_mb"] = round(self.baseline_memory_mb, 2)
        d["peak_memory_mb"] = round(self.peak_memory_mb, 2)
        d["memory_growth_mb"] = round(self.memory_growth_mb, 2)
        d["model_size_mb"] = round(self.model_size_mb, 2)
        return d


@dataclass(frozen=True)
class ModalityBenchmarkResult:
    """
    Unified benchmark evaluation result for a single medical modality / model.
    """

    modality: str
    model_name: str
    architecture: str
    classification_metrics: ClassificationMetrics
    performance_metrics: PerformanceMetrics
    composite_score: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modality": self.modality,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "composite_score": round(self.composite_score, 4),
            "timestamp": self.timestamp,
            "classification_metrics": self.classification_metrics.to_dict(),
            "performance_metrics": self.performance_metrics.to_dict(),
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# ----------------------------------------------------------------------
# Math Evaluation Functions
# ----------------------------------------------------------------------

def calculate_classification_metrics(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    y_prob: Optional[Union[List[List[float]], np.ndarray]] = None,
    class_names: Optional[List[str]] = None,
) -> ClassificationMetrics:
    """
    Calculate comprehensive multi-class classification diagnostic metrics.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True class labels (integers 0..K-1).
    y_pred : array-like of shape (n_samples,)
        Predicted class labels.
    y_prob : array-like of shape (n_samples, n_classes), optional
        Predicted class probabilities for ROC-AUC computation.
    class_names : List[str], optional
        Human-readable names for classes.

    Returns
    -------
    ClassificationMetrics
    """
    try:
        y_true_arr = np.asarray(y_true, dtype=int)
        y_pred_arr = np.asarray(y_pred, dtype=int)
        n_samples = len(y_true_arr)

        if n_samples == 0:
            raise MetricComputationError("Cannot compute metrics for 0 samples.")

        if len(y_pred_arr) != n_samples:
            raise MetricComputationError("Length mismatch between y_true and y_pred.")

        unique_classes = sorted(list(set(y_true_arr).union(set(y_pred_arr))))
        n_classes = len(unique_classes)

        if class_names is None:
            class_names = [f"Class_{c}" for c in unique_classes]

        # 1. Confusion Matrix
        cm = np.zeros((n_classes, n_classes), dtype=int)
        class_to_idx = {c: i for i, c in enumerate(unique_classes)}
        for t, p in zip(y_true_arr, y_pred_arr):
            cm[class_to_idx[t]][class_to_idx[p]] += 1

        # 2. Overall Accuracy
        correct = np.sum(y_true_arr == y_pred_arr)
        accuracy = float(correct / n_samples)

        # 3. Per-class & Macro Precision, Recall, F1
        per_class_f1: Dict[str, float] = {}
        precisions: List[float] = []
        recalls: List[float] = []
        f1s: List[float] = []

        for i, c in enumerate(unique_classes):
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            c_name = class_names[i] if i < len(class_names) else f"Class_{c}"
            per_class_f1[c_name] = f1

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = float(np.mean(f1s))

        # 4. ROC-AUC Computation (if probabilities provided)
        roc_auc: Optional[float] = None
        if y_prob is not None:
            try:
                y_prob_arr = np.asarray(y_prob, dtype=float)
                roc_auc = _calculate_roc_auc(y_true_arr, y_prob_arr, n_classes)
            except Exception:
                roc_auc = None

        return ClassificationMetrics(
            accuracy=accuracy,
            precision=macro_precision,
            recall=macro_recall,
            f1_score=macro_f1,
            roc_auc=roc_auc,
            confusion_matrix=cm.tolist(),
            n_samples=n_samples,
            class_names=class_names,
            per_class_f1=per_class_f1,
        )
    except Exception as exc:
        if isinstance(exc, MetricComputationError):
            raise
        raise MetricComputationError(f"Failed to calculate classification metrics: {exc}") from exc


def _calculate_roc_auc(y_true: np.ndarray, y_prob: np.ndarray, n_classes: int) -> float:
    """
    Calculate ROC-AUC score (binary or multi-class macro average).
    Pure NumPy implementation of trapezoidal integration under ROC curve.
    """
    if n_classes == 2 and y_prob.ndim == 1:
        return _binary_roc_auc(y_true, y_prob)
    elif n_classes == 2 and y_prob.ndim == 2 and y_prob.shape[1] == 2:
        return _binary_roc_auc(y_true, y_prob[:, 1])
    else:
        # Multi-class OvR (One-vs-Rest) macro-average AUC
        aucs = []
        for c in range(y_prob.shape[1]):
            binary_true = (y_true == c).astype(int)
            prob_c = y_prob[:, c]
            if len(np.unique(binary_true)) > 1:
                auc_c = _binary_roc_auc(binary_true, prob_c)
                aucs.append(auc_c)
        return float(np.mean(aucs)) if aucs else 0.5


def _binary_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Calculate binary ROC-AUC using rank sums (Mann-Whitney U statistic)."""
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort scores
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(len(y_score)) + 1

    # Handle ties in ranks
    unique_scores, counts = np.unique(y_score, return_counts=True)
    for score, count in zip(unique_scores, counts):
        if count > 1:
            idx = np.where(y_score == score)[0]
            ranks[idx] = np.mean(ranks[idx])

    pos_ranks_sum = np.sum(ranks[y_true == 1])
    u_stat = pos_ranks_sum - (n_pos * (n_pos + 1)) / 2.0
    auc = u_stat / (n_pos * n_neg)
    return float(np.clip(auc, 0.0, 1.0))


__all__ = [
    "ClassificationMetrics",
    "PerformanceMetrics",
    "ModalityBenchmarkResult",
    "calculate_classification_metrics",
]
