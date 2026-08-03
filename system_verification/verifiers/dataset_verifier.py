"""
system_verification/verifiers/dataset_verifier.py
===================================================
Verifies raw dataset integrity, file counts, modality coverage, corrupted images,
and ultrasound mask file contamination.

Author: Medical AI System — Module 5.5
"""

import json
import logging
from typing import Any, Dict, List

import pandas as pd

from system_verification.config import (
    M1_METADATA_CSV,
    M1_VALIDATION_REPORT_JSON,
    PATH_SAMPLE_SIZE,
    PROJECT_ROOT,
    SUPPORTED_MODALITIES,
)

logger = logging.getLogger(__name__)


class DatasetVerifier:
    """Verifies dataset integrity, modalities, corruption counts, and image path existence."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run dataset verification checks."""
        logger.info("Running DatasetVerifier ...")
        checks: List[Dict[str, Any]] = []

        total_scanned = 0
        corrupted_count = 0
        duplicate_group_count = 0

        # 1. Read Module 1 validation report
        if M1_VALIDATION_REPORT_JSON.exists():
            try:
                with open(M1_VALIDATION_REPORT_JSON, "r", encoding="utf-8") as fh:
                    v_rpt = json.load(fh)
                total_scanned = v_rpt.get("total_images_scanned", 0)
                corrupted_count = v_rpt.get("errors", 0)
                duplicate_group_count = v_rpt.get("warnings", 0)
                checks.append({
                    "check": "Validation Report JSON",
                    "status": "PASS",
                    "details": f"Scanned: {total_scanned}, Corrupted: {corrupted_count}, Duplicate Groups: {duplicate_group_count}",
                })
            except Exception as exc:
                checks.append({
                    "check": "Validation Report JSON",
                    "status": "FAIL",
                    "details": f"Error reading report: {exc}",
                })
        else:
            checks.append({
                "check": "Validation Report JSON",
                "status": "FAIL",
                "details": f"File not found at {M1_VALIDATION_REPORT_JSON}",
            })

        # 2. Inspect outputs/metadata.csv if present
        modality_counts: Dict[str, int] = {}
        mask_file_count = 0
        missing_sample_paths = 0

        if M1_METADATA_CSV.exists():
            try:
                df = pd.read_csv(M1_METADATA_CSV, low_memory=False)

                # Check modality coverage
                if "modality" in df.columns:
                    modality_counts = df["modality"].value_counts().to_dict()
                    found_modalities = set(modality_counts.keys())
                    missing_mods = SUPPORTED_MODALITIES - found_modalities
                    status = "PASS" if not missing_mods else "WARNING"
                    checks.append({
                        "check": "Modality Coverage",
                        "status": status,
                        "details": f"Found {len(found_modalities)}/{len(SUPPORTED_MODALITIES)} modalities. Missing: {missing_mods or 'None'}",
                    })

                # Check for mask files in image_id/image_path
                if "image_id" in df.columns:
                    mask_rows = df[df["image_id"].astype(str).str.contains("_mask", case=False, na=False)]
                    mask_file_count = len(mask_rows)
                    status = "WARNING" if mask_file_count > 0 else "PASS"
                    checks.append({
                        "check": "Ultrasound Mask Contamination",
                        "status": status,
                        "details": f"Detected {mask_file_count} mask files contaminating metadata",
                    })

                # Sample check image path accessibility
                if "image_path" in df.columns and len(df) > 0:
                    sample_df = df.sample(min(PATH_SAMPLE_SIZE, len(df)), random_state=42)
                    for p in sample_df["image_path"]:
                        abs_p = PROJECT_ROOT / p
                        if not abs_p.exists():
                            missing_sample_paths += 1
                    status = "PASS" if missing_sample_paths == 0 else "FAIL"
                    checks.append({
                        "check": "Raw Image Path Accessibility",
                        "status": status,
                        "details": f"Sampled {min(PATH_SAMPLE_SIZE, len(df))} paths, {missing_sample_paths} missing on disk",
                    })

            except Exception as exc:
                checks.append({
                    "check": "Module 1 metadata.csv Read",
                    "status": "FAIL",
                    "details": f"Error parsing CSV: {exc}",
                })
        else:
            checks.append({
                "check": "Module 1 metadata.csv Read",
                "status": "FAIL",
                "details": f"File not found at {M1_METADATA_CSV}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "total_images_scanned": total_scanned,
            "corrupted_count": corrupted_count,
            "duplicate_group_count": duplicate_group_count,
            "modality_counts": modality_counts,
            "mask_file_count": mask_file_count,
            "missing_sample_paths": missing_sample_paths,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("DatasetVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
