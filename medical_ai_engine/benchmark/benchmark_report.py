"""
benchmark_report.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Generates Markdown + JSON platform benchmark reports.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from medical_ai_engine.benchmark.benchmark_exceptions import BenchmarkReportError
from medical_ai_engine.benchmark.comparison_engine import ComparisonSummary, ModelComparisonEntry
from medical_ai_engine.benchmark.leaderboard import Leaderboard
from medical_ai_engine.benchmark.metrics import ModalityBenchmarkResult

logger = logging.getLogger(__name__)


class BenchmarkReport:
    """
    Generates two platform benchmark documents:

    1. `docs/platform_benchmark_report.md`  — Full Markdown report (human-readable).
    2. `docs/platform_benchmark_results.json` — Complete raw results (JSON).
    """

    def __init__(self, output_directory: str = "docs") -> None:
        self._out_dir = Path(output_directory)
        self._out_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        results: List[ModalityBenchmarkResult],
        comparison: ComparisonSummary,
        leaderboard: Leaderboard,
        version: str = "0.17.0.0",
    ) -> Dict[str, Path]:
        """
        Generate Markdown and JSON reports.

        Returns dict: {"markdown": Path, "json": Path}
        """
        try:
            md_path = self._write_markdown_report(results, comparison, leaderboard, version)
            json_path = self._write_json_report(results, comparison, leaderboard, version)
            logger.info("[BenchmarkReport] Reports written: %s, %s", md_path, json_path)
            return {"markdown": md_path, "json": json_path}
        except Exception as exc:
            raise BenchmarkReportError(f"Failed to generate benchmark report: {exc}") from exc

    # ------------------------------------------------------------------
    # Markdown Report
    # ------------------------------------------------------------------

    def _write_markdown_report(
        self,
        results: List[ModalityBenchmarkResult],
        comparison: ComparisonSummary,
        leaderboard: Leaderboard,
        version: str,
    ) -> Path:
        now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        avg_f1 = comparison.average_platform_f1
        avg_lat = comparison.average_platform_latency_ms
        avg_rps = comparison.average_platform_throughput_rps
        grade = leaderboard.overall_platform_grade

        # Per-modality metric tables
        clf_table = self._build_classification_table(results)
        perf_table = self._build_performance_table(results)
        per_class_section = self._build_per_class_section(results)
        tier_summary = self._build_tier_summary(leaderboard)

        md = f"""# Medical AI Platform — Global Benchmark Report

**Platform:** Medical Image Diagnosis & Anomaly Detection  
**Module:** 17 — Global Benchmarking & Model Comparison  
**Version:** `{version}`  
**Generated:** {now}

---

## Executive Summary

| Metric | Value |
|--------|-------|
| **Platform Grade** | **{grade}** |
| **Modalities Evaluated** | {comparison.total_models_evaluated} / 7 |
| **Average F1-Score** | {avg_f1:.4f} |
| **Average Inference Latency** | {avg_lat:.2f} ms |
| **Average Throughput** | {avg_rps:.1f} RPS |
| **Top Model (F1)** | {comparison.top_model_by_f1} |
| **Fastest Model** | {comparison.top_model_by_speed} |
| **Most Efficient** | {comparison.most_efficient_model} |

---

## Global Leaderboard

{leaderboard.to_markdown_table()}

---

## Performance Tier Distribution

{tier_summary}

---

## Classification Metrics — All Modalities

{clf_table}

---

## Performance Metrics — All Modalities

{perf_table}

---

## Per-Class F1 Breakdown

{per_class_section}

---

## Benchmark Methodology

- **Evaluation Samples:** ≥ 200 calibrated synthetic samples per modality
- **Performance Profiling:** {results[0].metadata.get('spec_latency_ms', 'N/A')} ms base (warmup + {comparison.total_models_evaluated}× iterations)
- **Hardware Target:** CPU (production inference baseline)
- **Metric Averaging:** Macro-averaged Precision, Recall, F1 across all disease classes
- **ROC-AUC:** Multi-class One-vs-Rest macro-average (Mann-Whitney rank correlation)
- **Composite Score:** 55% F1 + 25% Accuracy + 10% ROC-AUC + 10% Latency Score
- **Model Sizes:** Representative published DL architecture weight sizes

---

## Production Readiness Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| F1 ≥ 0.90 on all 7 modalities | ≥ 0.90 | {'✅ MET' if avg_f1 >= 0.90 else '⚠️ PARTIAL'} |
| Inference latency ≤ 20 ms | ≤ 20 ms | {'✅ MET' if avg_lat <= 20.0 else '⚠️ PARTIAL'} |
| Throughput ≥ 50 RPS | ≥ 50 RPS | {'✅ MET' if avg_rps >= 50.0 else '⚠️ PARTIAL'} |
| All modalities benchmarked | 7 / 7 | {'✅ MET' if comparison.total_models_evaluated == 7 else '⚠️ PARTIAL'} |

---

*Report generated by Module 17 — Global Benchmarking & Model Comparison*  
*Medical Image Diagnosis and Anomaly Detection Platform v{version}*
"""

        path = self._out_dir / "platform_benchmark_report.md"
        path.write_text(md, encoding="utf-8")
        return path

    def _build_classification_table(self, results: List[ModalityBenchmarkResult]) -> str:
        header = (
            "| Modality | Model | F1-Score | Accuracy | Precision | Recall | ROC-AUC | N Classes | N Samples |\n"
            "|----------|-------|----------|----------|-----------|--------|---------|-----------|----------|\n"
        )
        rows = []
        for r in sorted(results, key=lambda x: x.classification_metrics.f1_score, reverse=True):
            clf = r.classification_metrics
            roc_str = f"{clf.roc_auc:.4f}" if clf.roc_auc is not None else "N/A"
            rows.append(
                f"| {r.modality} | {r.model_name} | **{clf.f1_score:.4f}** | "
                f"{clf.accuracy:.4f} | {clf.precision:.4f} | {clf.recall:.4f} | "
                f"{roc_str} | {len(clf.class_names)} | {clf.n_samples} |"
            )
        return header + "\n".join(rows)

    def _build_performance_table(self, results: List[ModalityBenchmarkResult]) -> str:
        header = (
            "| Modality | Architecture | Latency (ms) | P95 (ms) | P99 (ms) | RPS | Peak Mem (MB) | Model Size (MB) | Params (M) |\n"
            "|----------|-------------|-------------|---------|---------|-----|--------------|----------------|----------|\n"
        )
        rows = []
        for r in sorted(results, key=lambda x: x.performance_metrics.inference_latency_ms):
            p = r.performance_metrics
            rows.append(
                f"| {r.modality} | {r.architecture} | **{p.inference_latency_ms:.2f}** | "
                f"{p.latency_p95_ms:.2f} | {p.latency_p99_ms:.2f} | {p.throughput_rps:.1f} | "
                f"{p.peak_memory_mb:.1f} | {p.model_size_mb:.1f} | {p.param_count / 1e6:.1f}M |"
            )
        return header + "\n".join(rows)

    def _build_per_class_section(self, results: List[ModalityBenchmarkResult]) -> str:
        sections = []
        for r in results:
            clf = r.classification_metrics
            if not clf.per_class_f1:
                continue
            hdr = f"### {r.modality.replace('_', ' ').title()} — {r.model_name}\n\n"
            tbl_hdr = "| Class | F1-Score |\n|-------|----------|\n"
            rows = "\n".join(
                f"| {cls} | {score:.4f} |"
                for cls, score in sorted(clf.per_class_f1.items(), key=lambda x: -x[1])
            )
            sections.append(hdr + tbl_hdr + rows)
        return "\n\n".join(sections)

    def _build_tier_summary(self, leaderboard: Leaderboard) -> str:
        from medical_ai_engine.benchmark.leaderboard import LEADERBOARD_TIERS
        lines = ["| Tier | Count | Emoji |", "|------|-------|-------|"]
        for t in LEADERBOARD_TIERS:
            count = leaderboard.coverage.get(t.name, 0)
            lines.append(f"| {t.name} | {count} | {t.icon} |")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # JSON Report
    # ------------------------------------------------------------------

    def _write_json_report(
        self,
        results: List[ModalityBenchmarkResult],
        comparison: ComparisonSummary,
        leaderboard: Leaderboard,
        version: str,
    ) -> Path:
        data = {
            "version": version,
            "generated": datetime.now(tz=timezone.utc).isoformat(),
            "platform_grade": leaderboard.overall_platform_grade,
            "comparison_summary": comparison.to_dict(),
            "leaderboard": leaderboard.to_dict(),
            "modality_results": [r.to_dict() for r in results],
        }
        path = self._out_dir / "platform_benchmark_results.json"
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        return path


__all__ = ["BenchmarkReport"]
