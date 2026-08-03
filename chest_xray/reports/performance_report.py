"""
chest_xray/reports/performance_report.py
=========================================
Performance report generator for Chest X-ray evaluation.
"""

from pathlib import Path
from typing import Any, Dict

def generate_performance_report_markdown(results: Dict[str, Any], output_path: str | Path) -> Path:
    from chest_xray.evaluation.evaluator import ChestXrayEvaluator
    # Evaluator handles markdown formatting inside save_reports
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
