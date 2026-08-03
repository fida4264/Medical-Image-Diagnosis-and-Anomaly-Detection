"""
run_augmentation.py
=====================
Entry point for Module 4: Medical Image Augmentation Framework.

Usage
-----
    python run_augmentation.py                   # augment all modalities
    python run_augmentation.py --sample          # process 5 per modality (testing)
    python run_augmentation.py --modality brain_mri
    python run_augmentation.py --factor 3        # 3 augmented copies per image
    python run_augmentation.py --seed 123        # custom random seed
    python run_augmentation.py --workers 4       # parallel threads

Pipeline Steps
--------------
[1/7]  Config     — Load configs/augmentation.yaml
[2/7]  Load       — Read preprocessing_metadata.csv from Module 3
[3/7]  Analyse    — Detect class imbalance, compute augmentation targets
[4/7]  Augment    — Apply modality-specific policies offline
[5/7]  Metadata   — Write augmentation_metadata.csv + augmentation_report.json
[6/7]  Statistics — Write augmentation_statistics.json
[7/7]  Preview    — Generate augmentation_preview/ comparison grids

Outputs
-------
augmented_images/{modality}/{class}/{id}_aug{N}.png
augmentation_metadata.csv           (inside augmented_images/)
augmentation_report.json            (inside augmented_images/)
augmentation_statistics.json        (inside augmented_images/)
augmentation_preview/{modality}_preview.png

Author: Medical AI System — Module 4
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging setup (must happen before module imports)
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent
LOGS_DIR = WORKSPACE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "augmentation.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("augmentation.run")

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from augmentation.config import (
    CONFIG_PATH,
    PROCESSED_IMAGES_DIR,
    PREPROCESSING_METADATA_CSV,
    MASTER_METADATA_CSV,
    AUGMENTED_IMAGES_DIR,
    AUGMENTATION_PREVIEW_DIR,
    AUGMENTATION_METADATA_CSV,
    AUGMENTATION_REPORT_JSON,
    AUGMENTATION_STATISTICS_JSON,
)
from augmentation.factory.augmentation_factory import AugmentationFactory
from augmentation.offline.offline_generator import OfflineGenerator
from augmentation.reports.metadata_writer import AugmentationMetadataWriter
from augmentation.reports.statistics import AugmentationStatistics
from augmentation.reports.preview_generator import PreviewGenerator

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

BANNER = """
+==============================================================================+
|                                                                              |
|   Medical Image Diagnosis and Anomaly Detection System                       |
|                                                                              |
|   Module 4 -- Medical Image Augmentation Framework                           |
|   Version 1.0.0                                                              |
|                                                                              |
+==============================================================================+
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{'=' * 78}\n  [{n}/{total}] {msg}\n{'=' * 78}")


def _ok(msg: str, elapsed: float) -> None:
    print(f"  [OK] {msg} ({elapsed:.2f}s)")


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load augmentation YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    logger.info("Config loaded: %s", config_path)
    return config


def load_metadata(sample_mode: bool, sample_size: int, modality_filter: Optional[str]) -> pd.DataFrame:
    """Load preprocessing metadata, falling back to master metadata."""
    if PREPROCESSING_METADATA_CSV.exists():
        df = pd.read_csv(PREPROCESSING_METADATA_CSV, encoding="utf-8")
        logger.info("Loaded preprocessing_metadata.csv: %d rows", len(df))
        # Filter to successful processed images
        if "status" in df.columns:
            df = df[df["status"].isin(["success", "skipped"])]
        path_col = "processed_path"
    elif MASTER_METADATA_CSV.exists():
        df = pd.read_csv(MASTER_METADATA_CSV, encoding="utf-8")
        logger.warning(
            "preprocessing_metadata.csv not found -- falling back to master_metadata.csv"
        )
        path_col = "image_path"
    else:
        raise FileNotFoundError(
            f"No metadata found. Expected:\n"
            f"  {PREPROCESSING_METADATA_CSV}\n"
            f"  {MASTER_METADATA_CSV}\n"
            "Run Module 3 (run_preprocessing.py) first."
        )

    if modality_filter:
        df = df[df["modality"] == modality_filter]
        logger.info("Filtered to modality '%s': %d rows", modality_filter, len(df))

    if sample_mode:
        df = (
            df.groupby("modality", group_keys=False)
            .apply(
                lambda g: g.sample(min(sample_size, len(g)), random_state=42),
                include_groups=False,
            )
        )
        logger.info("SAMPLE MODE: %d images (%d per modality)", len(df), sample_size)

    return df


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    sample_mode: bool = False,
    sample_size: int = 5,
    modality_filter: Optional[str] = None,
    augmentations_per_image_override: Optional[int] = None,
    seed_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
) -> int:
    """
    Execute the full Module 4 augmentation pipeline.

    Returns
    -------
    int
        0 on success, 1 on fatal error.
    """
    print(BANNER)
    total_steps = 7
    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load config
    # ------------------------------------------------------------------
    _step(1, total_steps, "Loading configs/augmentation.yaml ...")
    t0 = time.perf_counter()
    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        return 1

    general = config.get("general", {})
    n_aug = augmentations_per_image_override or int(general.get("augmentations_per_image", 5))
    seed = seed_override if seed_override is not None else int(general.get("random_seed", 42))
    num_workers = num_workers_override or int(general.get("num_workers", 8))
    skip_existing = bool(general.get("skip_existing", True))
    output_format = str(general.get("output_format", "png"))
    vis_cfg = config.get("visualization", {})
    samples_per_mod = int(vis_cfg.get("samples_per_modality", 5))
    vis_enabled = bool(vis_cfg.get("enabled", True))

    AUGMENTED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    AUGMENTATION_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    factory = AugmentationFactory(config)
    _ok(
        f"Config loaded | mode=offline | aug_per_img={n_aug} | seed={seed} | workers={num_workers}",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 2 — Load metadata
    # ------------------------------------------------------------------
    _step(2, total_steps, "Loading preprocessing metadata ...")
    t0 = time.perf_counter()
    try:
        df = load_metadata(sample_mode, sample_size, modality_filter)
    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        return 1

    n_modalities = df["modality"].nunique()
    _ok(f"{len(df):,} images | {n_modalities} modalities", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 3 — Analyse class distribution
    # ------------------------------------------------------------------
    _step(3, total_steps, "Analysing class distribution ...")
    t0 = time.perf_counter()

    generator = OfflineGenerator(config, factory)
    distribution = generator.analyze_class_distribution(df)
    targets = generator.compute_augmentation_targets(
        df, n_aug, config.get("class_balancing", {})
    )

    total_planned = sum(targets.values())
    for modality, class_counts in sorted(distribution.items()):
        print(f"  {modality}:")
        for cls, cnt in sorted(class_counts.items()):
            key = f"{modality}/{cls}"
            planned = targets.get(key, n_aug)
            planned_total = cnt * planned
            print(f"    {cls}: {cnt:,} originals -> {planned_total:,} augmented ({planned}x each)")

    _ok(f"Total augmented images planned: {total_planned:,}", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 4 — Offline augmentation
    # ------------------------------------------------------------------
    _step(4, total_steps, f"Generating augmented images ({num_workers} workers) ...")
    t0 = time.perf_counter()

    results = generator.run(
        metadata_df=df,
        output_dir=AUGMENTED_IMAGES_DIR,
        augmentations_per_image=n_aug,
        num_workers=num_workers,
        skip_existing=skip_existing,
        output_format=output_format,
        base_seed=seed,
        modality_filter=None,  # already filtered above
    )

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = len(results) - success - skipped
    _ok(
        f"Done: {success:,} augmented | {skipped:,} skipped | {errors:,} errors",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 5 — Write metadata & report
    # ------------------------------------------------------------------
    _step(5, total_steps, "Writing augmentation_metadata.csv + augmentation_report.json ...")
    t0 = time.perf_counter()

    metadata_writer = AugmentationMetadataWriter()
    metadata_writer.set_total_originals(len(df))
    for r in results:
        metadata_writer.record(
            original_image_id=r["original_image_id"],
            original_path=r["original_path"],
            augmented_path=r["augmented_path"],
            modality=r["modality"],
            normalized_class=r["normalized_class"],
            augmentation_index=r["augmentation_index"],
            operations_applied=r["operations_applied"],
            seed=r["seed"],
            status=r["status"],
        )

    csv_path = AUGMENTED_IMAGES_DIR / AUGMENTATION_METADATA_CSV
    report_path = AUGMENTED_IMAGES_DIR / AUGMENTATION_REPORT_JSON
    metadata_writer.save_csv(csv_path)
    metadata_writer.save_report(report_path)
    _ok(
        f"augmentation_metadata.csv: {metadata_writer.total_records:,} rows",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 6 — Compute statistics
    # ------------------------------------------------------------------
    _step(6, total_steps, "Computing augmentation_statistics.json ...")
    t0 = time.perf_counter()

    aug_metadata_df = pd.read_csv(csv_path, encoding="utf-8") if csv_path.exists() else pd.DataFrame()

    stats = AugmentationStatistics()
    stats.load_original(df)
    stats.load_augmented(aug_metadata_df)
    stats.compute()
    stats_path = AUGMENTED_IMAGES_DIR / AUGMENTATION_STATISTICS_JSON
    stats.save(stats_path)
    _ok("augmentation_statistics.json written", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 7 — Preview grids
    # ------------------------------------------------------------------
    _step(7, total_steps, "Generating augmentation preview grids ...")
    t0 = time.perf_counter()

    saved_previews = []
    if vis_enabled:
        preview_gen = PreviewGenerator(output_dir=AUGMENTATION_PREVIEW_DIR)

        # Pick one sample image per modality
        path_col = "processed_path" if "processed_path" in df.columns else "image_path"
        modality_image_map = {}
        augmenter_map = {}

        for modality in df["modality"].unique():
            mod_df = df[df["modality"] == modality]
            # Try to find an image that actually exists
            for _, row in mod_df.iterrows():
                img_path = Path(str(row.get(path_col, "")))
                if img_path.exists():
                    modality_image_map[modality] = img_path
                    if AugmentationFactory.is_supported(modality):
                        augmenter_map[modality] = factory.get(modality)
                    break

        saved_previews = preview_gen.generate_all_modalities(
            modality_image_map=modality_image_map,
            augmenter_map=augmenter_map,
            n_versions=samples_per_mod,
        )
    else:
        logger.info("Visualizations disabled in config.")

    _ok(f"{len(saved_previews)} preview grid(s) saved", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 78}")
    print("  MODULE 4 COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Augmented images dir  : {AUGMENTED_IMAGES_DIR}")
    print(f"  Preview dir           : {AUGMENTATION_PREVIEW_DIR}")
    print(f"  Original images       : {len(df):,}")
    print(f"  Augmented generated   : {success:,}")
    print(f"  Skipped (existing)    : {skipped:,}")
    print(f"  Errors                : {errors:,}")
    print(f"  Preview grids         : {len(saved_previews)}")
    print(f"  Total time            : {total_elapsed:.1f}s")
    print(f"{'=' * 78}\n")
    logger.info("Module 4 pipeline complete in %.1fs.", total_elapsed)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 4: Medical Image Augmentation Framework"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Process a small sample per modality for testing.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=5,
        help="Images per modality in sample mode (default: 5).",
    )
    parser.add_argument(
        "--modality", type=str, default=None,
        help="Augment only one modality (e.g., brain_mri, chest_xray).",
    )
    parser.add_argument(
        "--factor", type=int, default=None,
        help="Override augmentations per image (e.g., 3).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed.",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Override number of parallel workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    sys.exit(
        run_pipeline(
            sample_mode=args.sample,
            sample_size=args.sample_size,
            modality_filter=args.modality,
            augmentations_per_image_override=args.factor,
            seed_override=args.seed,
            num_workers_override=args.workers,
        )
    )
