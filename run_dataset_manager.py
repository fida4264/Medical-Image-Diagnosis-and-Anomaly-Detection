"""
run_dataset_manager.py
======================
Entry point for Module 1: Dataset Management.

Running this script executes the complete dataset management pipeline:

    1. Scan   — discovers all images across all modalities.
    2. Validate — checks integrity (corruption, duplicates, empty folders …).
    3. Metadata — enriches each record with dimensions, MD5, validity.
    4. Labels  — assigns integer labels and builds class distribution table.
    5. Stats   — computes and saves dataset_statistics.json.

Output files are written to the outputs/ directory:
    outputs/metadata.csv
    outputs/labels.csv
    outputs/class_distribution.csv
    outputs/dataset_statistics.json
    outputs/validation_report.json

Runtime logs are written to:
    logs/dataset_manager.log

Usage
-----
    python run_dataset_manager.py

Author: Medical AI System — Module 1
"""

import sys
import logging
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: ensure the project root is on sys.path so that
# `import dataset_manager` works regardless of the working directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_manager.config import (
    DATASETS_ROOT,
    LOG_DATE_FORMAT,
    LOG_FILE_NAME,
    LOG_FORMAT,
    LOG_LEVEL,
    LOGS_DIR,
    OUTPUTS_DIR,
)
from dataset_manager import (
    DatasetScanner,
    DatasetValidator,
    LabelGenerator,
    MetadataGenerator,
    StatisticsGenerator,
)


# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------

def _setup_logging() -> logging.Logger:
    """Configure root logger to write to both console and a log file."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOGS_DIR / LOG_FILE_NAME

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)   # full detail in file

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # concise on console

    root_logger = logging.getLogger()
    root_logger.setLevel(LOG_LEVEL)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return logging.getLogger("dataset_manager.run")


# ---------------------------------------------------------------------------
# Banner and Summary Helpers
# ---------------------------------------------------------------------------

def _print_banner() -> None:
    """Print a styled project banner to stdout."""
    banner = """
+==============================================================================+
|                                                                              |
|   Medical Image Diagnosis and Anomaly Detection System                       |
|                                                                              |
|   Module 1 -- Dataset Management                                             |
|   Version 1.0.0                                                              |
|                                                                              |
+==============================================================================+
"""
    print(banner)


def _print_section(title: str) -> None:
    """Print a section header."""
    width = 78
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def _print_summary_table(scanner, report, metadata_df, dist_df) -> None:
    """Print a comprehensive summary table to stdout."""
    import pandas as pd

    _print_section("DATASET SCAN SUMMARY")
    print(f"  Datasets root      : {DATASETS_ROOT}")
    print(f"  Total images found : {scanner.total_images:>8,}")
    print(f"  Modalities found   : {scanner.modality_count:>8}")
    print(f"  Skipped (hidden)   : {scanner.skipped_hidden:>8,}")
    print(f"  Skipped (unsupport): {scanner.skipped_unsupported:>8,}")

    _print_section("VALIDATION REPORT")
    print(f"  Images scanned     : {report.total_images_scanned:>8,}")
    print(f"  Total issues       : {report.total_issues:>8,}")
    print(f"  Errors             : {report.errors:>8,}")
    print(f"  Warnings           : {report.warnings:>8,}")
    if report.summary:
        print("  Issue breakdown:")
        for category, count in sorted(report.summary.items()):
            print(f"    +- {category:<28}: {count:>5}")

    _print_section("PER-MODALITY CLASS DISTRIBUTION")
    if dist_df is not None and not dist_df.empty:
        # Pivot for display
        pivot = dist_df.pivot_table(
            index=["modality", "disease_class"],
            columns="split",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        print(pivot.to_string())
    else:
        print("  No distribution data — datasets may be empty.")

    if metadata_df is not None and not metadata_df.empty:
        _print_section("IMAGE DIMENSION SUMMARY (valid images only)")
        valid = metadata_df[metadata_df["is_valid"] & (metadata_df["width"] > 0)]
        if not valid.empty:
            dim_stats = valid.groupby("modality")[["width", "height"]].agg(["min", "max", "mean"])
            dim_stats.columns = ["w_min", "w_max", "w_avg", "h_min", "h_max", "h_avg"]
            dim_stats = dim_stats.round(1)
            print(dim_stats.to_string())
        else:
            print("  No dimension data available.")

    _print_section("OUTPUT FILES")
    outputs = [
        OUTPUTS_DIR / "metadata.csv",
        OUTPUTS_DIR / "labels.csv",
        OUTPUTS_DIR / "class_distribution.csv",
        OUTPUTS_DIR / "dataset_statistics.json",
        OUTPUTS_DIR / "validation_report.json",
    ]
    for out in outputs:
        status = "[OK]" if out.exists() else "[MISSING]"
        size_kb = f"{out.stat().st_size / 1024:.1f} KB" if out.exists() else "---"
        print(f"  {status}  {out.name:<40}  {size_kb}")

    print(f"\n  Log file: {LOGS_DIR / LOG_FILE_NAME}")
    print()


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_pipeline() -> int:
    """
    Execute the full Module 1 dataset management pipeline.

    Returns
    -------
    int
        Exit code: 0 for success, 1 for fatal error.
    """
    log = _setup_logging()
    _print_banner()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    total_start = time.perf_counter()

    metadata_df = None
    dist_df = None
    report = None

    # ------------------------------------------------------------------
    # Step 1: Scan
    # ------------------------------------------------------------------
    _print_section("[1/5] Scanning datasets...")
    try:
        t0 = time.perf_counter()
        scanner = DatasetScanner(datasets_root=DATASETS_ROOT)
        records = scanner.scan()
        elapsed = time.perf_counter() - t0
        log.info("Scan complete in %.2f s — %d records", elapsed, len(records))
        print(f"  [OK] Found {len(records):,} images in {scanner.modality_count} modalities "
              f"({elapsed:.2f}s)")
    except FileNotFoundError as exc:
        log.error("Scan failed: %s", exc)
        print(f"\n  ❌  ERROR: {exc}")
        return 1

    if not records:
        log.warning(
            "No images were found in %s. "
            "Please add dataset folders before running this script.",
            DATASETS_ROOT,
        )
        print(
            f"\n  [WARNING] No images found in '{DATASETS_ROOT}'.\n"
            "     Place your dataset sub-folders inside the 'datasets/' directory.\n"
            "     The pipeline will still generate empty output files.\n"
        )

    # ------------------------------------------------------------------
    # Step 2: Validate
    # ------------------------------------------------------------------
    _print_section("[2/5] Validating dataset integrity...")
    try:
        t0 = time.perf_counter()
        validator = DatasetValidator(
            records=records,
            discovered_modalities=scanner.discovered_modalities,
            outputs_dir=OUTPUTS_DIR,
        )
        report = validator.validate()
        validator.save_report()
        elapsed = time.perf_counter() - t0
        log.info("Validation complete in %.2f s — %d issue(s)", elapsed, report.total_issues)
        print(f"  [OK] Validation done in {elapsed:.2f}s -- "
              f"{report.errors} error(s), {report.warnings} warning(s)")
    except Exception as exc:
        log.exception("Validation step failed: %s", exc)
        print(f"\n  [ERROR] Validation failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Step 3: Metadata
    # ------------------------------------------------------------------
    _print_section("[3/5] Generating metadata.csv...")
    try:
        t0 = time.perf_counter()
        meta_gen = MetadataGenerator(records=records, outputs_dir=OUTPUTS_DIR)
        metadata_df = meta_gen.generate()
        meta_gen.save()
        elapsed = time.perf_counter() - t0
        log.info("Metadata generated in %.2f s — %d rows", elapsed, len(metadata_df))
        print(f"  [OK] metadata.csv written in {elapsed:.2f}s  ({len(metadata_df):,} rows)")
    except Exception as exc:
        log.exception("Metadata generation failed: %s", exc)
        print(f"\n  [ERROR] Metadata generation failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Step 4: Labels + Class Distribution
    # ------------------------------------------------------------------
    _print_section("[4/5] Generating labels.csv and class_distribution.csv...")
    try:
        t0 = time.perf_counter()
        label_gen = LabelGenerator(metadata_df=metadata_df, outputs_dir=OUTPUTS_DIR)
        labels_df, dist_df = label_gen.generate()
        label_gen.save_labels()
        label_gen.save_class_distribution()
        elapsed = time.perf_counter() - t0
        log.info("Labels generated in %.2f s", elapsed)
        print(f"  [OK] labels.csv + class_distribution.csv written in {elapsed:.2f}s")
    except Exception as exc:
        log.exception("Label generation failed: %s", exc)
        print(f"\n  [ERROR] Label generation failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Step 5: Statistics
    # ------------------------------------------------------------------
    _print_section("[5/5] Computing dataset_statistics.json...")
    try:
        t0 = time.perf_counter()
        stats_gen = StatisticsGenerator(metadata_df=metadata_df, outputs_dir=OUTPUTS_DIR)
        stats_gen.compute()
        stats_gen.save()
        elapsed = time.perf_counter() - t0
        log.info("Statistics generated in %.2f s", elapsed)
        print(f"  [OK] dataset_statistics.json written in {elapsed:.2f}s")
    except Exception as exc:
        log.exception("Statistics generation failed: %s", exc)
        print(f"\n  [ERROR] Statistics generation failed: {exc}")
        return 1

    # ------------------------------------------------------------------
    # Final Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_start
    _print_summary_table(scanner, report, metadata_df, dist_df)

    log.info("Pipeline completed successfully in %.2f s.", total_elapsed)
    print(
        f"+============================================================+\n"
        f"|  [DONE] Module 1 complete -- {total_elapsed:.2f}s total                 |\n"
        f"+============================================================+\n"
    )
    return 0


# ---------------------------------------------------------------------------
# Script Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(run_pipeline())
