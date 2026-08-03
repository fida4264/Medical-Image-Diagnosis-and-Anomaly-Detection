"""
brain_ct/reports/performance_report.py
======================================
Performance report generator for Brain CT evaluation.
"""

from pathlib import Path
from typing import Any, Dict

def generate_performance_report_markdown(results: Dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
