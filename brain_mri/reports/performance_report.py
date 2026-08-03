"""
brain_mri/reports/performance_report.py
======================================
Consolidates performance statistics and writes summary reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


def write_performance_summary(
    eval_results: Dict[str, Any],
    output_path: str | Path = "saved_models/brain_mri/evaluation/performance_report.md",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    gm = eval_results.get("global_metrics", {})
    bm = eval_results.get("benchmark", {})

    lines = [
        "# Module 7: Brain MRI Model Performance Summary",
        "",
        f"- **Accuracy:** {gm.get('accuracy', 0.0)*100:.2f}%",
        f"- **Macro F1:** {gm.get('f1_macro', 0.0):.4f}",
        f"- **Weighted F1:** {gm.get('f1_weighted', 0.0):.4f}",
        f"- **Inference Speed:** {bm.get('throughput_fps', 0.0)} FPS",
        f"- **p50 Latency:** {bm.get('p50_latency_ms', 0.0)} ms",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Wrote performance summary to {path}")
    return path
