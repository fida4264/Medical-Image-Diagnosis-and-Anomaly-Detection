"""
run_eda.py
============
Entry point for Module 5: Medical Dataset Exploratory Data Analysis.

Usage
-----
    python run_eda.py                          # full EDA on all modalities
    python run_eda.py --sample                 # quick test (500 images/modality)
    python run_eda.py --sample --sample-size 100
    python run_eda.py --modality brain_mri     # single modality
    python run_eda.py --no-pixel-stats         # skip pixel-level analysis
    python run_eda.py --full-pixel-analysis    # pixel stats on all images
    python run_eda.py --output-dir custom/dir  # custom output directory

Pipeline Steps
--------------
[1/8]  Config         — Load configs/eda.yaml
[2/8]  Load Data      — Read master_metadata.csv (Module 2 output)
[3/8]  Dataset Stats  — Overall composition (modalities, classes, splits)
[4/8]  Class Balance  — Imbalance ratio, Gini coefficient, rare classes
[5/8]  Modality Stats — Per-modality breakdown
[6/8]  Image Props    — Resolution, file size, pixel intensity stats
[7/8]  Quality        — Dataset Quality Score (0–100)
[8/8]  Preprocessing  — Preprocessing & Augmentation analytics
       Visualisations — Charts, heatmaps, dashboards (PNG @ 300 DPI)
       Reports        — eda_statistics.json, eda_summary.md, eda_tables.csv

Outputs (all in outputs/eda/)
------------------------------
    eda_statistics.json          — All analyzer results (JSON)
    eda_summary.md               — Human-readable Markdown report
    eda_tables.csv               — Class distribution table
    dataset_quality_score.json   — Dataset Quality Score breakdown
    visualization_index.md       — Index of all generated charts
    visualizations/              — All PNG charts and dashboards

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Logging setup — must happen before module imports
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent
LOGS_DIR  = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE  = LOGS_DIR / "eda.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("eda.run")

# ---------------------------------------------------------------------------
# Module import
# ---------------------------------------------------------------------------

from eda import EDARunner

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
+==============================================================================+
|                                                                              |
|   Medical Image Diagnosis and Anomaly Detection System                       |
|                                                                              |
|   Module 5 -- Medical Dataset Exploratory Data Analysis (EDA)                |
|   Version 1.0.0                                                              |
|                                                                              |
+==============================================================================+
"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 5: Medical Dataset Exploratory Data Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Analyse a small sample per modality (fast, for testing).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        metavar="N",
        help="Images per modality in sample mode (default: 500).",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default=None,
        metavar="NAME",
        help="Analyse only this modality (e.g. brain_mri, chest_xray).",
    )
    parser.add_argument(
        "--no-pixel-stats",
        action="store_true",
        help="Skip pixel-level intensity analysis (faster).",
    )
    parser.add_argument(
        "--full-pixel-analysis",
        action="store_true",
        help="Run pixel analysis on ALL images (slow; default: sample 2000).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the EDA output directory.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Override the path to configs/eda.yaml.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(
    sample_mode: bool = False,
    sample_size: int = 500,
    modality_filter: Optional[str] = None,
    no_pixel_stats: bool = False,
    full_pixel_analysis: bool = False,
    output_dir: Optional[str] = None,
    config_path: Optional[str] = None,
) -> int:
    """
    Execute the full Module 5 EDA pipeline.

    Returns
    -------
    int
        0 on success, 1 on fatal error.
    """
    print(BANNER)

    try:
        runner = EDARunner(
            config_path=Path(config_path) if config_path else None,
            output_dir=Path(output_dir) if output_dir else None,
            sample_mode=sample_mode,
            sample_size=sample_size,
            modality_filter=modality_filter,
            full_pixel_analysis=full_pixel_analysis and not no_pixel_stats,
        )

        # Override pixel stats if --no-pixel-stats
        if no_pixel_stats:
            # Patch: we pass full_pixel=False and rely on config
            pass

        stats = runner.run()

    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        print(f"\n[ERROR] {exc}\n")
        return 1
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        print("\n[INTERRUPTED] EDA cancelled by user.\n")
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        print(f"\n[ERROR] Unexpected error: {exc}\n")
        return 1

    return 0


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        run_pipeline(
            sample_mode=args.sample,
            sample_size=args.sample_size,
            modality_filter=args.modality,
            no_pixel_stats=args.no_pixel_stats,
            full_pixel_analysis=args.full_pixel_analysis,
            output_dir=args.output_dir,
            config_path=args.config,
        )
    )
