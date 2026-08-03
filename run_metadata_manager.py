"""
run_metadata_manager.py
========================
Entry point for Module 2: Dataset Standardization & Metadata Management.

Running this script:
    python run_metadata_manager.py

Pipeline Steps
--------------
[1/6]  Load      — Read all Module 1 output files via MetadataLoader.
[2/6]  Standardize — Normalise modalities, extract true disease classes,
                     assign channels, align to master schema.
[3/6]  Map Classes — Build class_mapping.csv and modality_mapping.csv.
[4/6]  Map Images  — Generate unique internal IDs, build image_mapping.csv.
[5/6]  Registry   — Build dataset_registry.json.
[6/6]  Summarise  — Write dataset_summary.md and 5 visualisation charts.

Outputs (saved to metadata/)
-----------------------------
master_metadata.csv
image_mapping.csv
class_mapping.csv
modality_mapping.csv
dataset_registry.json
dataset_summary.md
visualizations/*.png (5 charts)

Logs
----
logs/metadata_manager.log

Author: Medical AI System — Module 2
"""

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Logging setup — must happen BEFORE any metadata_manager imports so that
# all sub-module loggers inherit this configuration.
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "metadata_manager.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("metadata_manager.run")

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

import pandas as pd

from metadata_manager.metadata_loader import MetadataLoader
from metadata_manager.metadata_standardizer import MetadataStandardizer
from metadata_manager.class_mapper import ClassMapper
from metadata_manager.image_mapper import ImageMapper
from metadata_manager.registry_generator import RegistryGenerator
from metadata_manager.summary_generator import SummaryGenerator
from metadata_manager.visualization import Visualizer
from metadata_manager.config import (
    METADATA_OUTPUT_DIR,
    MODALITY_NORMALIZE,
    MODALITY_ABBREVIATION,
    MODALITY_MAPPING_CSV,
)

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
+==============================================================================+
|                                                                              |
|   Medical Image Diagnosis and Anomaly Detection System                       |
|                                                                              |
|   Module 2 -- Dataset Standardization & Metadata Management                 |
|   Version 1.0.0                                                              |
|                                                                              |
+==============================================================================+
"""


def _print_step(n: int, total: int, msg: str) -> None:
    print(f"\n{'=' * 78}")
    print(f"  [{n}/{total}] {msg}")
    print(f"{'=' * 78}")


def _ok(msg: str, elapsed: float) -> None:
    print(f"  [OK] {msg} ({elapsed:.2f}s)")


def _save_modality_mapping(output_dir: Path) -> Path:
    """Save modality_mapping.csv mapping raw modality names to abbreviations."""
    rows = []
    for raw_mod, norm_mod in MODALITY_NORMALIZE.items():
        abbrev = MODALITY_ABBREVIATION.get(norm_mod, norm_mod.upper())
        rows.append({
            "raw_modality": raw_mod,
            "normalized_modality": norm_mod,
            "abbreviation": abbrev,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["normalized_modality"])
    dest = output_dir / MODALITY_MAPPING_CSV
    df.to_csv(dest, index=False, encoding="utf-8")
    logger.info("modality_mapping.csv saved -> %s  (%d rows)", dest, len(df))
    return dest


def run_pipeline() -> int:
    """
    Execute the full Module 2 pipeline.

    Returns
    -------
    int
        0 on success, 1 on failure.
    """
    print(BANNER)
    total_steps = 6
    pipeline_start = time.perf_counter()
    output_dir = METADATA_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load Module 1 outputs
    # ------------------------------------------------------------------
    _print_step(1, total_steps, "Loading Module 1 outputs ...")
    t0 = time.perf_counter()
    try:
        loader = MetadataLoader()
        loader.load()
    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        print(f"\n  [ERROR] {exc}")
        return 1

    meta_df = loader.metadata_df
    statistics = loader.statistics
    _ok(loader.summary(), time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 2 — Standardize metadata
    # ------------------------------------------------------------------
    _print_step(2, total_steps, "Standardizing metadata ...")
    t0 = time.perf_counter()
    class_mapper = ClassMapper()
    standardizer = MetadataStandardizer(meta_df, class_mapper, output_dir)
    master_df = standardizer.standardize()
    elapsed = time.perf_counter() - t0
    _ok(
        f"master_metadata shape: {master_df.shape[0]:,} rows x {master_df.shape[1]} cols",
        elapsed,
    )

    # ------------------------------------------------------------------
    # Step 3 — Class mapping + modality mapping
    # ------------------------------------------------------------------
    _print_step(3, total_steps, "Generating class_mapping.csv and modality_mapping.csv ...")
    t0 = time.perf_counter()

    # Class mapping
    raw_classes = master_df["original_class"].dropna().unique().tolist()
    if "disease" in master_df.columns:
        raw_classes += master_df["disease"].dropna().unique().tolist()
    class_mapper.build_mapping(list(set(raw_classes)))
    class_mapper.save(output_dir)

    # Modality mapping
    _save_modality_mapping(output_dir)

    n_classes = class_mapper.mapping_df["normalized_class"].nunique() if class_mapper.mapping_df is not None else 0
    _ok(
        f"class_mapping: {len(raw_classes)} raw variants -> {n_classes} normalized classes",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 4 — Image mapping (internal IDs)
    # ------------------------------------------------------------------
    _print_step(4, total_steps, "Generating image_mapping.csv (internal IDs) ...")
    t0 = time.perf_counter()
    image_mapper = ImageMapper(master_df, output_dir)
    id_series = image_mapper.generate_ids()
    standardizer.assign_internal_ids(id_series)
    image_mapper.save(output_dir)

    # Now save master_metadata.csv (with IDs filled in)
    master_path = standardizer.save(output_dir)
    _ok(
        f"image_mapping: {len(id_series):,} IDs | master_metadata.csv written",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 5 — Dataset registry
    # ------------------------------------------------------------------
    _print_step(5, total_steps, "Generating dataset_registry.json ...")
    t0 = time.perf_counter()
    # Re-read master_df so registry sees the final version (with internal_id)
    master_df_final = standardizer.master_df if standardizer.master_df is not None else master_df
    registry_gen = RegistryGenerator(master_df_final, statistics, output_dir)
    registry = registry_gen.generate()
    registry_gen.save(output_dir)
    _ok(
        f"Registry: {registry['total_datasets']} dataset entries",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 6 — Summary + visualisations
    # ------------------------------------------------------------------
    _print_step(6, total_steps, "Generating dataset_summary.md and visualisations ...")
    t0 = time.perf_counter()

    summary_gen = SummaryGenerator(master_df_final, statistics, output_dir)
    summary_gen.generate()
    summary_gen.save(output_dir)

    visualizer = Visualizer(master_df_final, output_dir / "visualizations")
    charts = visualizer.generate_all()
    _ok(
        f"Summary written | {len(charts)} chart(s) saved",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Final summary table
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 78}")
    print("  MODULE 2 COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output directory  : {output_dir}")
    print(f"  master_metadata   : {master_df_final.shape[0]:,} rows")
    print(f"  Modalities        : {master_df_final['modality'].nunique()}")
    if "normalized_class" in master_df_final.columns:
        print(f"  Normalized classes: {master_df_final['normalized_class'].nunique()}")
    print(f"  Charts saved      : {len(charts)}")
    print(f"  Total time        : {total_elapsed:.1f}s")
    print(f"{'=' * 78}\n")
    logger.info("Module 2 pipeline complete in %.1fs.", total_elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(run_pipeline())
