"""
system_verification/verifiers/preprocessing_verifier.py
=========================================================
Verifies Module 3 preprocessing outputs, execution coverage, quality report flag rates,
and processed image file accessibility.

Author: Medical AI System — Module 5.5
"""

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from system_verification.config import (
    M3_PREPROCESSING_METADATA_CSV,
    M3_PREPROCESSING_REPORT_JSON,
    M3_QUALITY_REPORT_JSON,
    MODULE3_OUTPUTS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


class PreprocessingVerifier:
    """Verifies preprocessing outputs, execution coverage, and quality flag statistics."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run preprocessing verification checks."""
        logger.info("Running PreprocessingVerifier ...")
        checks: List[Dict[str, Any]] = []

        processed_count = 0
        quality_flagged = 0
        quality_flag_rate = 0.0
        by_modality_quality: Dict[str, Any] = {}
        missing_processed_files = 0

        # 1. Preprocessing metadata CSV
        if M3_PREPROCESSING_METADATA_CSV.exists():
            try:
                df = pd.read_csv(M3_PREPROCESSING_METADATA_CSV, low_memory=False)
                processed_count = len(df)
                status = "PASS" if processed_count > 0 else "WARNING"
                checks.append({
                    "check": "Preprocessing Metadata CSV",
                    "status": status,
                    "details": f"Processed images recorded in CSV: {processed_count}",
                })

                # Check execution coverage vs total dataset (~118,715)
                coverage_pct = round((processed_count / 118715) * 100, 3)
                cov_status = "PASS" if coverage_pct >= 90 else "WARNING"
                checks.append({
                    "check": "Preprocessing Execution Coverage",
                    "status": cov_status,
                    "details": f"Coverage: {processed_count}/118,715 ({coverage_pct}% executed)",
                })

                # Spot-check image existence on disk
                if "processed_path" in df.columns and len(df) > 0:
                    for p in df["processed_path"].head(50):
                        abs_p = PROJECT_ROOT / p
                        if not abs_p.exists():
                            missing_processed_files += 1
                    status = "PASS" if missing_processed_files == 0 else "FAIL"
                    checks.append({
                        "check": "Processed Image File Existence",
                        "status": status,
                        "details": f"Sampled 50 processed images, missing on disk: {missing_processed_files}",
                    })

            except Exception as exc:
                checks.append({
                    "check": "Preprocessing Metadata CSV Read",
                    "status": "FAIL",
                    "details": f"Failed to read CSV: {exc}",
                })
        else:
            checks.append({
                "check": "Preprocessing Metadata CSV Existence",
                "status": "WARNING",
                "details": f"File not found at {M3_PREPROCESSING_METADATA_CSV}",
            })

        # 2. Quality report JSON
        if M3_QUALITY_REPORT_JSON.exists():
            try:
                with open(M3_QUALITY_REPORT_JSON, "r", encoding="utf-8") as fh:
                    q_rpt = json.load(fh)
                quality_flagged = q_rpt.get("total_flagged", 0)
                quality_flag_rate = q_rpt.get("flag_rate_pct", 0.0)
                by_modality_quality = q_rpt.get("per_modality", {})

                # High flag rate warning (> 30% flag rate is aggressive)
                flag_status = "WARNING" if quality_flag_rate > 30.0 else "PASS"
                checks.append({
                    "check": "Image Quality Flag Rate",
                    "status": flag_status,
                    "details": f"Flagged: {quality_flagged}/{q_rpt.get('total_images', 0)} ({quality_flag_rate}% flag rate)",
                })
            except Exception as exc:
                checks.append({
                    "check": "Quality Report JSON",
                    "status": "FAIL",
                    "details": f"Error parsing JSON: {exc}",
                })
        else:
            checks.append({
                "check": "Quality Report JSON Existence",
                "status": "WARNING",
                "details": f"File not found at {M3_QUALITY_REPORT_JSON}",
            })

        # 3. Preprocessing report JSON
        if M3_PREPROCESSING_REPORT_JSON.exists():
            checks.append({
                "check": "Preprocessing Summary Report JSON",
                "status": "PASS",
                "details": f"File exists at {M3_PREPROCESSING_REPORT_JSON.name}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "processed_count": processed_count,
            "quality_flagged": quality_flagged,
            "quality_flag_rate": quality_flag_rate,
            "by_modality_quality": by_modality_quality,
            "missing_processed_files": missing_processed_files,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("PreprocessingVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
