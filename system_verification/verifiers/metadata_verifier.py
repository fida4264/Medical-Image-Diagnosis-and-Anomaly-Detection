"""
system_verification/verifiers/metadata_verifier.py
====================================================
Verifies master_metadata.csv schema integrity, null value counts, duplicate IDs,
class imbalance metrics, and split distribution.

Author: Medical AI System — Module 5.5
"""

import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from system_verification.config import (
    M2_DATASET_REGISTRY_JSON,
    M2_MASTER_METADATA_CSV,
    M2_VISUALIZATIONS_DIR,
    MASTER_METADATA_REQUIRED_COLUMNS,
    SUPPORTED_MODALITIES,
)

logger = logging.getLogger(__name__)


def _gini_coefficient(counts: np.ndarray) -> float:
    """Compute Gini coefficient of array (0 = perfect balance, 1 = total inequality)."""
    if len(counts) == 0 or counts.sum() == 0:
        return 0.0
    c = np.sort(counts.astype(float))
    n = len(c)
    idx = np.arange(1, n + 1)
    return round(float(np.clip((2 * (idx * c).sum()) / (n * c.sum()) - (n + 1) / n, 0, 1)), 4)


class MetadataVerifier:
    """Verifies metadata schema, duplicate IDs, null values, and class balance."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run metadata verification checks."""
        logger.info("Running MetadataVerifier ...")
        checks: List[Dict[str, Any]] = []

        total_records = 0
        null_count = 0
        dup_image_ids = 0
        dup_internal_ids = 0
        unknown_class_count = 0
        gini_coeff = 0.0
        imbalance_ratio = 0.0
        has_md5 = False

        if M2_MASTER_METADATA_CSV.exists():
            try:
                df = pd.read_csv(M2_MASTER_METADATA_CSV, low_memory=False)
                total_records = len(df)

                # 1. Schema check
                missing_cols = set(MASTER_METADATA_REQUIRED_COLUMNS) - set(df.columns)
                status = "PASS" if not missing_cols else "FAIL"
                checks.append({
                    "check": "Master Metadata Schema",
                    "status": status,
                    "details": f"Missing columns: {missing_cols or 'None'}",
                })

                # Check optional md5_hash propagation
                has_md5 = "md5_hash" in df.columns
                checks.append({
                    "check": "MD5 Hash Column Propagation",
                    "status": "PASS" if has_md5 else "WARNING",
                    "details": "md5_hash column is present" if has_md5 else "md5_hash column missing from master_metadata.csv (BUG-3)",
                })

                # 2. Null values
                null_count = int(df.isnull().sum().sum())
                checks.append({
                    "check": "Zero Null Values",
                    "status": "PASS" if null_count == 0 else "FAIL",
                    "details": f"Total null fields found: {null_count}",
                })

                # 3. Duplicate IDs
                if "image_id" in df.columns:
                    dup_image_ids = int(df["image_id"].duplicated().sum())
                if "internal_id" in df.columns:
                    dup_internal_ids = int(df["internal_id"].duplicated().sum())

                checks.append({
                    "check": "Unique Image IDs",
                    "status": "PASS" if dup_image_ids == 0 else "FAIL",
                    "details": f"Duplicate image_ids: {dup_image_ids}",
                })
                checks.append({
                    "check": "Unique Internal IDs",
                    "status": "PASS" if dup_internal_ids == 0 else "FAIL",
                    "details": f"Duplicate internal_ids: {dup_internal_ids}",
                })

                # 4. Unknown class check & Gini coefficient
                if "normalized_class" in df.columns:
                    class_counts = df["normalized_class"].value_counts()
                    unknown_class_count = int((df["normalized_class"] == "unknown").sum())
                    gini_coeff = _gini_coefficient(class_counts.values)

                    max_c = class_counts.iloc[0] if len(class_counts) > 0 else 0
                    min_c = class_counts.iloc[-1] if len(class_counts) > 0 else 0
                    imbalance_ratio = round(max_c / min_c, 2) if min_c > 0 else 0.0

                    unknown_pct = (unknown_class_count / total_records) * 100 if total_records > 0 else 0
                    status = "WARNING" if unknown_pct > 10 else "PASS"
                    checks.append({
                        "check": "Class Label Completeness",
                        "status": status,
                        "details": f"Unknown class count: {unknown_class_count} ({unknown_pct:.1f}% of dataset)",
                    })

                # 5. Split distribution
                if "split" in df.columns:
                    splits = df["split"].value_counts().to_dict()
                    unspecified_pct = (splits.get("unspecified", 0) / total_records) * 100 if total_records > 0 else 0
                    checks.append({
                        "check": "Split Allocation",
                        "status": "WARNING" if unspecified_pct > 50 else "PASS",
                        "details": f"Splits: {splits} ({unspecified_pct:.1f}% unspecified)",
                    })

            except Exception as exc:
                checks.append({
                    "check": "Master Metadata CSV Read",
                    "status": "FAIL",
                    "details": f"Failed to load CSV: {exc}",
                })
        else:
            checks.append({
                "check": "Master Metadata CSV Existence",
                "status": "FAIL",
                "details": f"File not found at {M2_MASTER_METADATA_CSV}",
            })

        # 6. Dataset Registry JSON
        registry_count = 0
        if M2_DATASET_REGISTRY_JSON.exists():
            try:
                with open(M2_DATASET_REGISTRY_JSON, "r", encoding="utf-8") as fh:
                    reg = json.load(fh)
                registry_count = len(reg)
                checks.append({
                    "check": "Dataset Registry JSON",
                    "status": "PASS",
                    "details": f"Registered datasets: {registry_count}",
                })
            except Exception as exc:
                checks.append({
                    "check": "Dataset Registry JSON",
                    "status": "FAIL",
                    "details": f"Error parsing JSON: {exc}",
                })
        else:
            checks.append({
                "check": "Dataset Registry JSON",
                "status": "FAIL",
                "details": f"File not found at {M2_DATASET_REGISTRY_JSON}",
            })

        # 7. Metadata Visualizations
        vis_count = 0
        if M2_VISUALIZATIONS_DIR.exists():
            vis_files = list(M2_VISUALIZATIONS_DIR.glob("*.png"))
            vis_count = len(vis_files)
            checks.append({
                "check": "Metadata Visualizations",
                "status": "PASS" if vis_count >= 4 else "WARNING",
                "details": f"Generated visualization charts: {vis_count}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "total_records": total_records,
            "null_count": null_count,
            "dup_image_ids": dup_image_ids,
            "dup_internal_ids": dup_internal_ids,
            "unknown_class_count": unknown_class_count,
            "gini_coefficient": gini_coeff,
            "imbalance_ratio": imbalance_ratio,
            "registry_count": registry_count,
            "visualizations_count": vis_count,
            "has_md5": has_md5,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("MetadataVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
