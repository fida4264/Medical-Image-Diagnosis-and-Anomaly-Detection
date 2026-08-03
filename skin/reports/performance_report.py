"""
skin/reports/performance_report.py
==================================
Performance report generator for Module 12.
Generates human-readable Markdown and JSON performance summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class PerformanceReportGenerator:
    """Generates structured performance reports for model evaluation."""

    @staticmethod
    def generate(eval_results: Dict[str, Any], output_dir: str | Path = "outputs/skin") -> None:
        """Generate JSON and Markdown performance reports."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_path = output_dir / "performance_report.json"
        md_path = output_dir / "performance_report.md"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, default=str)

        metrics = eval_results.get("global_metrics", {})
        bm = eval_results.get("benchmark", {})

        md_content = f"""# Module 12: Skin Lesion Cancer Classification — Performance Report

## Executive Summary

- **Total Test Samples:** {eval_results.get('num_samples', 0)}
- **Test Accuracy:** {metrics.get('accuracy', 0)*100:.2f}%
- **Macro F1 Score:** {metrics.get('f1_macro', 0):.4f}
- **Weighted F1 Score:** {metrics.get('f1_weighted', 0):.4f}
- **Mean Inference Latency:** {bm.get('mean_latency_ms', 0):.2f} ms
- **Throughput:** {bm.get('fps_throughput', 0):.1f} FPS

## Class Distribution

Supported Skin Lesion Classes:
{', '.join(eval_results.get('class_names', []))}

---
*Report generated automatically by Module 12 PerformanceReportGenerator.*
"""

        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Performance reports written to '{json_path}' and '{md_path}'.")
