"""
system_verification/verifiers/eda_verifier.py
===============================================
Verifies Module 5 EDA output directory, eda_statistics.json, eda_summary.md,
dataset_quality_score.json, and visualization charts.

Author: Medical AI System — Module 5.5
"""

import json
import logging
from typing import Any, Dict, List

from system_verification.config import (
    M5_DATASET_QUALITY_SCORE_JSON,
    M5_EDA_STATISTICS_JSON,
    M5_EDA_SUMMARY_MD,
    M5_EDA_TABLES_CSV,
    M5_VISUALIZATION_INDEX_MD,
    M5_VISUALIZATIONS_DIR,
    MODULE5_OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


class EDAVerifier:
    """Verifies Module 5 output files, quality score JSON, and EDA visualizations."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run EDA verification checks."""
        logger.info("Running EDAVerifier ...")
        checks: List[Dict[str, Any]] = []

        eda_dir_exists = MODULE5_OUTPUTS_DIR.exists()
        quality_score = 0.0
        quality_grade = "N/A"
        visualization_count = 0

        checks.append({
            "check": "EDA Output Directory Existence",
            "status": "PASS" if eda_dir_exists else "WARNING",
            "details": f"Directory exists: {eda_dir_exists} (Module 5 execution status)",
        })

        if M5_EDA_STATISTICS_JSON.exists():
            checks.append({
                "check": "EDA Statistics JSON",
                "status": "PASS",
                "details": f"File exists at {M5_EDA_STATISTICS_JSON.name}",
            })
        else:
            checks.append({
                "check": "EDA Statistics JSON",
                "status": "WARNING",
                "details": f"File not found at {M5_EDA_STATISTICS_JSON} — Module 5 not executed at scale",
            })

        if M5_EDA_SUMMARY_MD.exists():
            checks.append({
                "check": "EDA Summary MD",
                "status": "PASS",
                "details": f"File exists at {M5_EDA_SUMMARY_MD.name}",
            })
        else:
            checks.append({
                "check": "EDA Summary MD",
                "status": "WARNING",
                "details": f"File not found at {M5_EDA_SUMMARY_MD}",
            })

        if M5_DATASET_QUALITY_SCORE_JSON.exists():
            try:
                with open(M5_DATASET_QUALITY_SCORE_JSON, "r", encoding="utf-8") as fh:
                    q_data = json.load(fh)
                quality_score = q_data.get("total_score", 0.0)
                quality_grade = q_data.get("grade", "N/A")
                checks.append({
                    "check": "Dataset Quality Score JSON",
                    "status": "PASS",
                    "details": f"Score: {quality_score}/100 ({quality_grade})",
                })
            except Exception as exc:
                checks.append({
                    "check": "Dataset Quality Score JSON Read",
                    "status": "FAIL",
                    "details": f"Failed to parse JSON: {exc}",
                })
        else:
            checks.append({
                "check": "Dataset Quality Score JSON",
                "status": "WARNING",
                "details": f"File not found at {M5_DATASET_QUALITY_SCORE_JSON}",
            })

        if M5_VISUALIZATIONS_DIR.exists():
            vis_files = list(M5_VISUALIZATIONS_DIR.glob("*.png"))
            visualization_count = len(vis_files)
            status = "PASS" if visualization_count > 0 else "WARNING"
            checks.append({
                "check": "EDA Visualizations Directory",
                "status": status,
                "details": f"Generated charts found: {visualization_count}",
            })
        else:
            checks.append({
                "check": "EDA Visualizations Directory",
                "status": "WARNING",
                "details": f"Directory not found at {M5_VISUALIZATIONS_DIR}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "dir_exists": eda_dir_exists,
            "quality_score": quality_score,
            "quality_grade": quality_grade,
            "visualization_count": visualization_count,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("EDAVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
