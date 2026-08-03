"""
validator.py
============
Dataset integrity validation for the Medical AI System.

This module takes the list of ImageRecord objects produced by DatasetScanner
and performs a thorough series of checks to identify data quality issues
before any model training or evaluation begins.

Checks Performed
----------------
1. Corrupted Images     — PIL cannot fully decode the pixel data.
2. Duplicate Images     — Two or more files share identical MD5 hashes.
3. Empty Folders        — A disease / split folder contains zero images.
4. Oversized Images     — File larger than MAX_FILE_SIZE_BYTES (warning only).
5. Undersized Images    — File smaller than MIN_FILE_SIZE_BYTES (treated as corrupted).
6. Missing Modalities   — An expected modality from KNOWN_MODALITIES was not found.
7. Single-class Split   — A split (train/val/test) contains only one class.

All findings are collected into a ValidationReport and serialised to JSON.

Classes
-------
ValidationIssue     Dataclass representing one validation finding.
ValidationReport    Aggregated collection of all issues with summary statistics.
DatasetValidator    Main validation orchestrator.

Author: Medical AI System — Module 1
"""

import json
import logging
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dataset_manager.config import (
    KNOWN_MODALITIES,
    MAX_FILE_SIZE_BYTES,
    MIN_FILE_SIZE_BYTES,
    OUTPUTS_DIR,
    VALIDATION_REPORT_JSON,
)
from dataset_manager.image_utils import compute_md5_hash, is_corrupted
from dataset_manager.scanner import ImageRecord

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class ValidationIssue:
    """
    Represents a single data-quality finding from the validation pass.

    Attributes
    ----------
    severity : str
        "ERROR" for issues that will definitely harm training/evaluation.
        "WARNING" for issues that may or may not be problems.
        "INFO" for informational notices.
    category : str
        Short category tag, e.g. "CORRUPTED", "DUPLICATE", "EMPTY_FOLDER".
    description : str
        Human-readable explanation of the issue.
    affected_path : str
        File or folder path (as string) that triggered the issue.
    modality : str
        Modality name, if applicable; empty string otherwise.
    disease_class : str
        Disease class name, if applicable; empty string otherwise.
    """
    severity: str
    category: str
    description: str
    affected_path: str
    modality: str = ""
    disease_class: str = ""


@dataclass
class ValidationReport:
    """
    Aggregated result of the full validation pass.

    Attributes
    ----------
    total_images_scanned : int
        Total number of ImageRecords that were inspected.
    total_issues : int
        Total number of issues found (all severities combined).
    errors : int
        Count of ERROR-severity issues.
    warnings : int
        Count of WARNING-severity issues.
    issues : list of ValidationIssue
        All individual issues, ordered by severity then category.
    duplicate_groups : dict
        Maps MD5 hash → list of file paths sharing that hash.
    summary : dict
        Key counts for quick access without iterating all issues.
    """
    total_images_scanned: int = 0
    total_issues: int = 0
    errors: int = 0
    warnings: int = 0
    issues: List[ValidationIssue] = field(default_factory=list)
    duplicate_groups: Dict[str, List[str]] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)

    def add_issue(self, issue: ValidationIssue) -> None:
        """Append an issue and update severity counters."""
        self.issues.append(issue)
        self.total_issues += 1
        if issue.severity == "ERROR":
            self.errors += 1
        elif issue.severity == "WARNING":
            self.warnings += 1

    def to_dict(self) -> dict:
        """Convert the report to a plain dictionary (JSON-serialisable)."""
        return {
            "total_images_scanned": self.total_images_scanned,
            "total_issues": self.total_issues,
            "errors": self.errors,
            "warnings": self.warnings,
            "summary": self.summary,
            "duplicate_groups": self.duplicate_groups,
            "issues": [asdict(i) for i in self.issues],
        }


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class DatasetValidator:
    """
    Validates a list of ImageRecord objects and produces a ValidationReport.

    Parameters
    ----------
    records : list of ImageRecord
        The full list of discovered image records from DatasetScanner.
    discovered_modalities : list of str, optional
        The modality names actually found during scanning (used to check for
        missing modalities against KNOWN_MODALITIES).

    Examples
    --------
    >>> validator = DatasetValidator(records, discovered_modalities=["brain_mri"])
    >>> report = validator.validate()
    >>> validator.save_report()
    """

    def __init__(
        self,
        records: List[ImageRecord],
        discovered_modalities: Optional[List[str]] = None,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._records = records
        self._discovered_modalities = discovered_modalities or []
        self._outputs_dir = outputs_dir or OUTPUTS_DIR
        self._report = ValidationReport()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def validate(self) -> ValidationReport:
        """
        Run all validation checks and return the completed ValidationReport.

        Returns
        -------
        ValidationReport
            Contains all detected issues and summary statistics.
        """
        logger.info("Starting dataset validation on %d records …", len(self._records))
        self._report = ValidationReport(total_images_scanned=len(self._records))

        self._check_corrupted_images()
        self._check_duplicates()
        self._check_empty_folders()
        self._check_oversized_images()
        self._check_missing_modalities()
        self._check_single_class_splits()
        self._build_summary()

        logger.info(
            "Validation complete. Issues: %d (Errors: %d | Warnings: %d)",
            self._report.total_issues,
            self._report.errors,
            self._report.warnings,
        )
        return self._report

    def save_report(self, output_path: Optional[Path] = None) -> Path:
        """
        Serialise the ValidationReport to a JSON file.

        Parameters
        ----------
        output_path : Path, optional
            Custom output path; defaults to OUTPUTS_DIR/validation_report.json.

        Returns
        -------
        Path
            The path where the report was written.
        """
        dest = output_path or (self._outputs_dir / VALIDATION_REPORT_JSON)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(self._report.to_dict(), fh, indent=2, default=str)
        logger.info("Validation report saved → %s", dest)
        return dest

    # ------------------------------------------------------------------
    # Individual Checks
    # ------------------------------------------------------------------

    def _check_corrupted_images(self) -> None:
        """Flag images that PIL cannot decode — runs checks in parallel."""
        logger.debug("  Checking for corrupted images (parallel) ...")
        paths = [rec.image_path for rec in self._records]
        rec_map = {rec.image_path: rec for rec in self._records}
        count = 0

        workers = min(os.cpu_count() or 4, 8)
        # ThreadPoolExecutor is safe for I/O-bound PIL checks
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_path = {pool.submit(is_corrupted, p): p for p in paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    bad = future.result()
                except Exception:
                    bad = True
                if bad:
                    rec = rec_map[path]
                    self._report.add_issue(ValidationIssue(
                        severity="ERROR",
                        category="CORRUPTED",
                        description=(
                            f"Image cannot be decoded or is too small "
                            f"(< {MIN_FILE_SIZE_BYTES} bytes)."
                        ),
                        affected_path=str(path),
                        modality=rec.modality,
                        disease_class=rec.disease_class,
                    ))
                    count += 1
        logger.debug("    -> %d corrupted image(s) found.", count)

    def _check_duplicates(self) -> None:
        """Detect images that share identical MD5 hashes (exact duplicates) — parallel."""
        logger.debug("  Checking for duplicate images (parallel) ...")
        paths = [rec.image_path for rec in self._records]
        hash_map: Dict[str, List[str]] = defaultdict(list)

        workers = min(os.cpu_count() or 4, 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            future_to_path = {pool.submit(compute_md5_hash, p): p for p in paths}
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    digest = future.result()
                except Exception:
                    digest = None
                if digest:
                    hash_map[digest].append(str(path))

        duplicate_count = 0
        for digest, dup_paths in hash_map.items():
            if len(dup_paths) > 1:
                self._report.duplicate_groups[digest] = dup_paths
                duplicate_count += 1
                self._report.add_issue(ValidationIssue(
                    severity="WARNING",
                    category="DUPLICATE",
                    description=(
                        f"Found {len(dup_paths)} files with identical MD5 ({digest[:8]}...). "
                        f"Duplicates: {', '.join(dup_paths)}"
                    ),
                    affected_path=dup_paths[0],
                ))
        logger.debug("    -> %d duplicate group(s) found.", duplicate_count)

    def _check_empty_folders(self) -> None:
        """
        Identify disease / split folders that contain zero valid images.

        Uses the (modality, disease_class, split) combinations present in the
        record list to determine which folders were successfully scanned; then
        walks the filesystem to find folders that exist but produced no records.
        """
        logger.debug("  Checking for empty folders …")
        from dataset_manager.config import DATASETS_ROOT
        from dataset_manager.image_utils import is_supported_image

        populated: set = {
            (rec.modality, rec.disease_class, rec.split)
            for rec in self._records
        }

        # Walk every leaf directory inside the datasets root
        count = 0
        for folder in DATASETS_ROOT.rglob("*"):
            if not folder.is_dir() or folder.name.startswith("."):
                continue
            # Check if this folder contains ANY image files directly
            has_images = any(
                f.is_file() and is_supported_image(f)
                for f in folder.iterdir()
            )
            if not has_images:
                # Check if it contains sub-folders (then it's a parent, not a leaf)
                has_subdirs = any(d.is_dir() for d in folder.iterdir())
                if not has_subdirs:
                    # Leaf folder with no images
                    self._report.add_issue(ValidationIssue(
                        severity="WARNING",
                        category="EMPTY_FOLDER",
                        description="Folder exists but contains no supported images.",
                        affected_path=str(folder),
                    ))
                    count += 1
        logger.debug("    → %d empty folder(s) found.", count)

    def _check_oversized_images(self) -> None:
        """Flag images exceeding the configured maximum file size threshold."""
        logger.debug("  Checking for oversized images …")
        count = 0
        for rec in self._records:
            if rec.file_size_bytes > MAX_FILE_SIZE_BYTES:
                self._report.add_issue(ValidationIssue(
                    severity="WARNING",
                    category="OVERSIZED",
                    description=(
                        f"File size {rec.file_size_bytes / 1_000_000:.1f} MB "
                        f"exceeds the {MAX_FILE_SIZE_BYTES / 1_000_000:.0f} MB threshold."
                    ),
                    affected_path=str(rec.image_path),
                    modality=rec.modality,
                    disease_class=rec.disease_class,
                ))
                count += 1
        logger.debug("    → %d oversized image(s) found.", count)

    def _check_missing_modalities(self) -> None:
        """Warn about expected modalities absent from the datasets directory."""
        logger.debug("  Checking for missing modalities …")
        found = {m.lower() for m in self._discovered_modalities}
        for expected in KNOWN_MODALITIES:
            if expected.lower() not in found:
                self._report.add_issue(ValidationIssue(
                    severity="WARNING",
                    category="MISSING_MODALITY",
                    description=(
                        f"Expected modality folder '{expected}' was not found. "
                        "Download the dataset and place it in the datasets/ directory."
                    ),
                    affected_path=str(
                        Path("datasets") / expected
                    ),
                    modality=expected,
                ))
        logger.debug("    → Check complete.")

    def _check_single_class_splits(self) -> None:
        """
        Warn about any modality/split combination containing only one disease class.

        A model trained on a single class cannot learn to distinguish between
        classes, making the split useless for supervised learning.
        """
        logger.debug("  Checking for single-class splits …")
        split_classes: Dict[tuple, set] = defaultdict(set)

        for rec in self._records:
            key = (rec.modality, rec.split)
            split_classes[key].add(rec.disease_class)

        for (modality, split), classes in split_classes.items():
            if len(classes) == 1:
                self._report.add_issue(ValidationIssue(
                    severity="WARNING",
                    category="SINGLE_CLASS_SPLIT",
                    description=(
                        f"Split '{split}' for modality '{modality}' contains "
                        f"only one class: {list(classes)[0]}. "
                        "Cannot train a classification model on a single class."
                    ),
                    affected_path=str(Path("datasets") / modality),
                    modality=modality,
                ))
        logger.debug("    → Check complete.")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(self) -> None:
        """Compute per-category issue counts for the report summary."""
        category_counts: Dict[str, int] = defaultdict(int)
        for issue in self._report.issues:
            category_counts[issue.category] += 1
        self._report.summary = dict(category_counts)
