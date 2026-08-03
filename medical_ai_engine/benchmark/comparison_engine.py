"""
comparison_engine.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Comparative analytics, trade-off analysis, and relative rank scoring engine.
"""

from __future__ import annotations

import numpy as np

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from medical_ai_engine.benchmark.benchmark_exceptions import ComparisonEngineError
from medical_ai_engine.benchmark.metrics import ModalityBenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class ModelComparisonEntry:
    """Detailed comparative record for a single modality/model."""

    rank: int
    modality: str
    model_name: str
    architecture: str
    composite_score: float
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    roc_auc: Optional[float]
    inference_latency_ms: float
    throughput_rps: float
    peak_memory_mb: float
    model_size_mb: float
    param_count: int
    efficiency_ratio: float  # F1 per ms of latency
    delta_f1_vs_top: float = 0.0
    delta_latency_vs_top: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "modality": self.modality,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "composite_score": round(self.composite_score, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "peak_memory_mb": round(self.peak_memory_mb, 2),
            "model_size_mb": round(self.model_size_mb, 2),
            "param_count": self.param_count,
            "efficiency_ratio": round(self.efficiency_ratio, 4),
            "delta_f1_vs_top": round(self.delta_f1_vs_top, 4),
            "delta_latency_vs_top": round(self.delta_latency_vs_top, 3),
        }


@dataclass
class ComparisonSummary:
    """Summary of platform-wide model comparisons across all modalities."""

    entries: List[ModelComparisonEntry]
    top_model_by_f1: str
    top_model_by_speed: str
    most_efficient_model: str
    average_platform_f1: float
    average_platform_latency_ms: float
    average_platform_throughput_rps: float
    total_models_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top_model_by_f1": self.top_model_by_f1,
            "top_model_by_speed": self.top_model_by_speed,
            "most_efficient_model": self.most_efficient_model,
            "average_platform_f1": round(self.average_platform_f1, 4),
            "average_platform_latency_ms": round(self.average_platform_latency_ms, 3),
            "average_platform_throughput_rps": round(self.average_platform_throughput_rps, 2),
            "total_models_evaluated": self.total_models_evaluated,
            "entries": [e.to_dict() for e in self.entries],
        }


class ComparisonEngine:
    """
    Ranks, analyzes, and compares benchmark results across modalities and models.

    Weighted Composite Score Formula
    --------------------------------
    CompositeScore = w_f1 * F1 + w_acc * Accuracy + w_lat * LatencyScore + w_mem * MemoryScore
    """

    def __init__(
        self,
        weight_f1: float = 0.40,
        weight_acc: float = 0.30,
        weight_latency: float = 0.15,
        weight_memory: float = 0.15,
    ) -> None:
        self.w_f1 = weight_f1
        self.w_acc = weight_acc
        self.w_lat = weight_latency
        self.w_mem = weight_memory

    def compare_results(
        self,
        results: List[ModalityBenchmarkResult],
    ) -> ComparisonSummary:
        """
        Rank and analyze a collection of ModalityBenchmarkResult objects.

        Parameters
        ----------
        results : List[ModalityBenchmarkResult]
            Evaluation results for each modality / model.

        Returns
        -------
        ComparisonSummary
        """
        if not results:
            raise ComparisonEngineError("Cannot run comparison engine on empty benchmark results list.")

        # Compute normalized composite scores
        max_lat = max(r.performance_metrics.inference_latency_ms for r in results) or 1.0
        max_mem = max(r.performance_metrics.peak_memory_mb for r in results) or 1.0

        entries_with_score: List[Tuple[float, ModalityBenchmarkResult, float]] = []

        for r in results:
            f1 = r.classification_metrics.f1_score
            acc = r.classification_metrics.accuracy
            lat = r.performance_metrics.inference_latency_ms
            mem = r.performance_metrics.peak_memory_mb

            # Latency and memory score: 1.0 is best (fastest / lowest memory)
            lat_score = max(0.0, 1.0 - (lat / (max_lat * 2.0)))
            mem_score = max(0.0, 1.0 - (mem / (max_mem * 2.0)))

            comp_score = (
                self.w_f1 * f1
                + self.w_acc * acc
                + self.w_lat * lat_score
                + self.w_mem * mem_score
            )
            efficiency = f1 / max(0.001, lat)
            entries_with_score.append((comp_score, r, efficiency))

        # Sort descending by composite score
        entries_with_score.sort(key=lambda x: x[0], reverse=True)

        top_entry = entries_with_score[0][1]
        top_f1 = top_entry.classification_metrics.f1_score
        top_lat = top_entry.performance_metrics.inference_latency_ms

        comparison_entries: List[ModelComparisonEntry] = []
        for rank, (comp_score, r, eff) in enumerate(entries_with_score, start=1):
            c_f1 = r.classification_metrics.f1_score
            c_lat = r.performance_metrics.inference_latency_ms

            entry = ModelComparisonEntry(
                rank=rank,
                modality=r.modality,
                model_name=r.model_name,
                architecture=r.architecture,
                composite_score=comp_score,
                f1_score=c_f1,
                accuracy=r.classification_metrics.accuracy,
                precision=r.classification_metrics.precision,
                recall=r.classification_metrics.recall,
                roc_auc=r.classification_metrics.roc_auc,
                inference_latency_ms=c_lat,
                throughput_rps=r.performance_metrics.throughput_rps,
                peak_memory_mb=r.performance_metrics.peak_memory_mb,
                model_size_mb=r.performance_metrics.model_size_mb,
                param_count=r.performance_metrics.param_count,
                efficiency_ratio=eff,
                delta_f1_vs_top=c_f1 - top_f1,
                delta_latency_vs_top=c_lat - top_lat,
            )
            comparison_entries.append(entry)

        # Platform aggregations
        by_f1 = max(results, key=lambda r: r.classification_metrics.f1_score)
        by_speed = min(results, key=lambda r: r.performance_metrics.inference_latency_ms)
        by_eff = max(comparison_entries, key=lambda e: e.efficiency_ratio)

        avg_f1 = float(np.mean([r.classification_metrics.f1_score for r in results]))
        avg_lat = float(np.mean([r.performance_metrics.inference_latency_ms for r in results]))
        avg_rps = float(np.mean([r.performance_metrics.throughput_rps for r in results]))

        return ComparisonSummary(
            entries=comparison_entries,
            top_model_by_f1=f"{by_f1.modality} ({by_f1.model_name})",
            top_model_by_speed=f"{by_speed.modality} ({by_speed.model_name})",
            most_efficient_model=f"{by_eff.modality} ({by_eff.model_name})",
            average_platform_f1=avg_f1,
            average_platform_latency_ms=avg_lat,
            average_platform_throughput_rps=avg_rps,
            total_models_evaluated=len(results),
        )


__all__ = ["ComparisonEngine", "ModelComparisonEntry", "ComparisonSummary"]
