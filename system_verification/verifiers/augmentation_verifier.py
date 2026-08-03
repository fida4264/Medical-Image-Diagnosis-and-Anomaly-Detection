"""
system_verification/verifiers/augmentation_verifier.py
========================================================
Verifies Module 4 offline augmentation outputs, metadata CSV, JSON reports, and
augmentation factors.

Author: Medical AI System — Module 5.5
"""

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from system_verification.config import (
    M4_AUGMENTATION_METADATA_CSV,
    M4_AUGMENTATION_REPORT_JSON,
    M4_AUGMENTATION_STATISTICS_JSON,
    MODULE4_OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


class AugmentationVerifier:
    """Verifies Module 4 output directory, augmentation reports, and output metadata."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run augmentation verification checks."""
        logger.info("Running AugmentationVerifier ...")
        checks: List[Dict[str, Any]] = []

        augmented_count = 0
        augmentation_factor = 0.0
        success_rate_pct = 0.0
        dir_exists = MODULE4_OUTPUTS_DIR.exists()

        checks.append({
            "check": "Augmented Images Directory Existence",
            "status": "PASS" if dir_exists else "WARNING",
            "details": f"Directory exists: {dir_exists} (Module 4 execution status)",
        })

        if M4_AUGMENTATION_METADATA_CSV.exists():
            try:
                df = pd.read_csv(M4_AUGMENTATION_METADATA_CSV, low_memory=False)
                augmented_count = len(df)
                status = "PASS" if augmented_count > 0 else "WARNING"
                checks.append({
                    "check": "Augmentation Metadata CSV",
                    "status": status,
                    "details": f"Augmented records in CSV: {augmented_count}",
                })
            except Exception as exc:
                checks.append({
                    "check": "Augmentation Metadata CSV Read",
                    "status": "FAIL",
                    "details": f"Failed to read CSV: {exc}",
                })
        else:
            checks.append({
                "check": "Augmentation Metadata CSV",
                "status": "WARNING",
                "details": f"File not found at {M4_AUGMENTATION_METADATA_CSV} — Module 4 not executed at scale",
            })

        if M4_AUGMENTATION_REPORT_JSON.exists():
            try:
                with open(M4_AUGMENTATION_REPORT_JSON, "r", encoding="utf-8") as fh:
                    a_rpt = json.load(fh)
                augmentation_factor = a_rpt.get("augmentation_factor", 0.0)
                success_rate_pct = a_rpt.get("success_rate_pct", 0.0)
                checks.append({
                    "check": "Augmentation Report JSON",
                    "status": "PASS",
                    "details": f"Factor: {augmentation_factor}x, Success Rate: {success_rate_pct}%",
                })
            except Exception as exc:
                checks.append({
                    "check": "Augmentation Report JSON Read",
                    "status": "FAIL",
                    "details": f"Failed to parse JSON: {exc}",
                })
        else:
            checks.append({
                "check": "Augmentation Report JSON",
                "status": "WARNING",
                "details": f"File not found at {M4_AUGMENTATION_REPORT_JSON}",
            })

        if M4_AUGMENTATION_STATISTICS_JSON.exists():
            checks.append({
                "check": "Augmentation Statistics JSON",
                "status": "PASS",
                "details": f"File exists at {M4_AUGMENTATION_STATISTICS_JSON.name}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "dir_exists": dir_exists,
            "augmented_count": augmented_count,
            "augmentation_factor": augmentation_factor,
            "success_rate_pct": success_rate_pct,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("AugmentationVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
