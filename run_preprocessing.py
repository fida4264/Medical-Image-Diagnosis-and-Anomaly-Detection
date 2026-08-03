"""
run_preprocessing.py
======================
Entry point for Module 3: Medical Image Preprocessing Framework.

Usage
-----
    python run_preprocessing.py              # process all 118,715 images
    python run_preprocessing.py --sample     # process 10 per modality (testing)
    python run_preprocessing.py --modality brain_mri     # single modality
    python run_preprocessing.py --workers 4              # override thread count

Pipeline Steps
--------------
[1/6]  Load      — Read master_metadata.csv from Module 2.
[2/6]  Configure — Load configs/preprocessing.yaml.
[3/6]  Process   — Apply modality-specific pipeline to every image (parallel).
[4/6]  Quality   — Write quality_report.json.
[5/6]  Metadata  — Write preprocessing_metadata.csv + preprocessing_report.json.
[6/6]  Visualize — Save before/after comparison grids.

Outputs (all inside processed_images/)
---------------------------------------
{modality}/{normalized_class}/{internal_id}.png
preprocessing_metadata.csv
preprocessing_report.json
quality_report.json
before_after/{modality}_sample.png  (5 per modality)

Author: Medical AI System — Module 3
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Logging (must happen before any preprocessing imports)
# ---------------------------------------------------------------------------

LOGS_DIR = Path(__file__).resolve().parent / "logs"
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "preprocessing.log"

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("preprocessing.run")

# ---------------------------------------------------------------------------
# Module imports
# ---------------------------------------------------------------------------

from preprocessing.common.image_loader import ImageLoader
from preprocessing.common.image_validator import ImageValidator
from preprocessing.common.image_saver import ImageSaver
from preprocessing.factory.preprocessor_factory import PreprocessorFactory
from preprocessing.reports.quality_report import QualityReportWriter
from preprocessing.reports.metadata_writer import MetadataWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WORKSPACE = Path(__file__).resolve().parent
METADATA_CSV = WORKSPACE / "metadata" / "master_metadata.csv"
CONFIG_PATH = WORKSPACE / "configs" / "preprocessing.yaml"
OUTPUT_DIR = WORKSPACE / "processed_images"

BANNER = """
+==============================================================================+
|                                                                              |
|   Medical Image Diagnosis and Anomaly Detection System                       |
|                                                                              |
|   Module 3 -- Medical Image Preprocessing Framework                         |
|   Version 1.0.0                                                              |
|                                                                              |
+==============================================================================+
"""


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and return the preprocessing YAML configuration."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    logger.info("Config loaded: %s", config_path)
    return config


# ---------------------------------------------------------------------------
# Single image processing function (called from threads)
# ---------------------------------------------------------------------------

def process_one_image(
    row: pd.Series,
    factory: PreprocessorFactory,
    quality_config: Dict,
    output_dir: Path,
    output_format: str,
    skip_existing: bool,
) -> Dict[str, Any]:
    """
    Process a single image row from master_metadata.csv.

    Returns a dict with all fields needed by MetadataWriter and
    QualityReportWriter.
    """
    image_id = str(row.get("image_id", ""))
    internal_id = str(row.get("internal_id", ""))
    modality = str(row.get("modality", ""))
    norm_class = str(row.get("normalized_class", "unknown"))
    src_path = str(row.get("image_path", ""))

    # Determine output path
    ext = f".{output_format.lstrip('.')}"
    out_path = output_dir / modality / norm_class / f"{internal_id}{ext}"

    # Skip if already processed
    if skip_existing and out_path.exists():
        return {
            "image_id": image_id,
            "internal_id": internal_id,
            "original_path": src_path,
            "processed_path": str(out_path),
            "modality": modality,
            "normalized_class": norm_class,
            "operations_applied": [],
            "original_shape": (0, 0),
            "processed_shape": (224, 224, 3),
            "processing_time_s": 0.0,
            "status": "skipped",
            "quality_flags": [],
        }

    t0 = time.perf_counter()
    status = "success"
    operations: List[str] = []
    quality_flags: List[str] = []
    original_shape = (0, 0)
    processed_shape = (224, 224, 3)

    try:
        # Load
        image = ImageLoader.load(src_path)
        original_shape = image.shape

        # Quality check (pre-processing)
        quality_flags = ImageValidator.validate_with_config(image, quality_config)

        # Get preprocessor and run pipeline
        preprocessor = factory.get(modality)
        processed, operations = preprocessor.preprocess(image)
        processed_shape = processed.shape

        # Save
        ImageSaver.save(processed, out_path)

    except FileNotFoundError as exc:
        status = f"error:file_not_found"
        logger.warning("[SKIP] %s: %s", internal_id, exc)
    except ValueError as exc:
        status = f"error:unsupported_modality"
        logger.warning("[SKIP] %s: %s", internal_id, exc)
    except OSError as exc:
        status = f"error:io"
        logger.warning("[SKIP] %s: %s", internal_id, exc)
    except Exception as exc:
        status = f"error:{type(exc).__name__}"
        logger.error("[ERROR] %s: %s", internal_id, exc)

    elapsed = time.perf_counter() - t0

    return {
        "image_id": image_id,
        "internal_id": internal_id,
        "original_path": src_path,
        "processed_path": str(out_path) if status == "success" else "",
        "modality": modality,
        "normalized_class": norm_class,
        "operations_applied": operations,
        "original_shape": original_shape,
        "processed_shape": processed_shape,
        "processing_time_s": elapsed,
        "status": status,
        "quality_flags": quality_flags,
    }


# ---------------------------------------------------------------------------
# Visualization — before/after comparison grids
# ---------------------------------------------------------------------------

def _make_comparison_grid(
    samples: List[Dict],
    modality: str,
    output_dir: Path,
    n_cols: int = 5,
) -> Optional[Path]:
    """Generate a before/after comparison grid for one modality."""
    if not samples:
        return None

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    fig.patch.set_facecolor("#1A1A2E")
    fig.suptitle(
        f"{modality.replace('_', ' ').title()} — Before / After Preprocessing",
        color="white", fontsize=14, fontweight="bold", y=0.98,
    )

    for col, rec in enumerate(samples):
        src_path = rec.get("original_path", "")
        dst_path = rec.get("processed_path", "")

        # Load original
        try:
            orig = cv2.cvtColor(ImageLoader.load(src_path), cv2.COLOR_BGR2RGB)
        except Exception:
            orig = np.zeros((224, 224, 3), dtype=np.uint8)

        # Load processed
        try:
            proc = cv2.cvtColor(ImageLoader.load(dst_path), cv2.COLOR_BGR2RGB)
        except Exception:
            proc = np.zeros((224, 224, 3), dtype=np.uint8)

        for row_idx, (ax, img, label) in enumerate(
            [(axes[0][col], orig, "Original"), (axes[1][col], proc, "Processed")]
        ):
            ax.imshow(img, cmap="gray" if img.mean(axis=2).std() < 5 else None)
            ax.set_title(label if col == 0 else "", color="white", fontsize=9)
            ax.axis("off")
            ax.set_facecolor("#1A1A2E")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    dest = output_dir / f"{modality}_sample.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(dest, dpi=120, bbox_inches="tight", facecolor="#1A1A2E")
    plt.close()
    logger.info("  Visualization saved: %s", dest.name)
    return dest


def generate_visualizations(
    results: List[Dict],
    output_dir: Path,
    samples_per_modality: int = 5,
) -> List[Path]:
    """Generate before/after grids for every modality that had successes."""
    logger.info("Generating before/after visualizations ...")
    vis_dir = output_dir / "before_after"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Group successful results by modality
    by_modality: Dict[str, List[Dict]] = {}
    for rec in results:
        if rec["status"] == "success" and rec["processed_path"]:
            mod = rec["modality"]
            by_modality.setdefault(mod, []).append(rec)

    saved = []
    for modality, recs in sorted(by_modality.items()):
        sample = recs[: min(samples_per_modality, len(recs))]
        path = _make_comparison_grid(sample, modality, vis_dir)
        if path:
            saved.append(path)

    logger.info("Visualizations complete: %d grids saved.", len(saved))
    return saved


# ---------------------------------------------------------------------------
# Progress counter (tqdm-free)
# ---------------------------------------------------------------------------

class _ProgressCounter:
    def __init__(self, total: int) -> None:
        self.total = total
        self._done = 0
        self._lock = __import__("threading").Lock()
        self._t0 = time.perf_counter()

    def increment(self) -> None:
        with self._lock:
            self._done += 1
            if self._done % 500 == 0 or self._done == self.total:
                elapsed = time.perf_counter() - self._t0
                rate = self._done / elapsed if elapsed > 0 else 0
                eta = (self.total - self._done) / rate if rate > 0 else 0
                pct = self._done / self.total * 100
                print(
                    f"  Progress: {self._done:,}/{self.total:,} "
                    f"({pct:.1f}%) | {rate:.0f} img/s | ETA: {eta:.0f}s",
                    flush=True,
                )


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, msg: str) -> None:
    print(f"\n{'=' * 78}\n  [{n}/{total}] {msg}\n{'=' * 78}")


def _ok(msg: str, elapsed: float) -> None:
    print(f"  [OK] {msg} ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    sample_mode: bool = False,
    sample_size: int = 10,
    modality_filter: Optional[str] = None,
    num_workers_override: Optional[int] = None,
) -> int:
    """
    Execute the full Module 3 preprocessing pipeline.

    Returns
    -------
    int
        0 on success, 1 on fatal error.
    """
    print(BANNER)
    total_steps = 6
    pipeline_start = time.perf_counter()

    # ------------------------------------------------------------------
    # Step 1 — Load metadata
    # ------------------------------------------------------------------
    _step(1, total_steps, "Loading master_metadata.csv from Module 2 ...")
    t0 = time.perf_counter()

    if not METADATA_CSV.exists():
        logger.error("FATAL: %s not found. Run Module 2 first.", METADATA_CSV)
        return 1

    df = pd.read_csv(METADATA_CSV, encoding="utf-8")
    logger.info("Loaded %d rows from master_metadata.csv", len(df))

    # Filter by modality if requested
    if modality_filter:
        df = df[df["modality"] == modality_filter]
        logger.info("Filtered to modality '%s': %d images", modality_filter, len(df))

    # Sample mode
    if sample_mode:
        df = (
            df.groupby("modality", group_keys=False)
            .apply(lambda g: g.sample(min(sample_size, len(g)), random_state=42), include_groups=False)
        )
        # Re-add modality column (excluded by include_groups=False in newer pandas)
        if "modality" not in df.columns:
            df = df.reset_index()  # fallback
        logger.info("SAMPLE MODE: %d images (%d per modality)", len(df), sample_size)

    _ok(f"{len(df):,} images to process across {df['modality'].nunique()} modalities", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 2 — Load config
    # ------------------------------------------------------------------
    _step(2, total_steps, "Loading preprocessing configuration ...")
    t0 = time.perf_counter()

    try:
        config = load_config(CONFIG_PATH)
    except FileNotFoundError as exc:
        logger.error("FATAL: %s", exc)
        return 1

    general = config.get("general", {})
    num_workers = num_workers_override or int(general.get("num_workers", 8))
    skip_existing = bool(general.get("skip_existing", True))
    output_format = str(general.get("output_format", "png"))
    quality_config = config.get("quality", {})
    vis_cfg = config.get("visualization", {})

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    factory = PreprocessorFactory(config)

    _ok(f"Config loaded | workers={num_workers} | skip_existing={skip_existing}", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Step 3 — Process images (parallel)
    # ------------------------------------------------------------------
    _step(3, total_steps, f"Processing {len(df):,} images with {num_workers} workers ...")
    t0 = time.perf_counter()

    rows = list(df.itertuples(index=False, name="Row"))
    quality_writer = QualityReportWriter()
    metadata_writer = MetadataWriter()
    progress = _ProgressCounter(len(rows))
    all_results: List[Dict] = []

    def _process(row) -> Dict:
        result = process_one_image(
            row._asdict(),  # type: ignore[attr-defined]
            factory,
            quality_config,
            OUTPUT_DIR,
            output_format,
            skip_existing,
        )
        progress.increment()
        return result

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(_process, row): row for row in rows}
        for future in as_completed(futures):
            try:
                result = future.result()
                all_results.append(result)
                quality_writer.record(
                    result["internal_id"],
                    result["modality"],
                    result["quality_flags"],
                )
                metadata_writer.record(
                    image_id=result["image_id"],
                    internal_id=result["internal_id"],
                    original_path=result["original_path"],
                    processed_path=result["processed_path"],
                    modality=result["modality"],
                    normalized_class=result["normalized_class"],
                    operations_applied=result["operations_applied"],
                    original_shape=result["original_shape"],
                    processed_shape=result["processed_shape"],
                    processing_time_s=result["processing_time_s"],
                    status=result["status"],
                    quality_flags=result["quality_flags"],
                )
            except Exception as exc:
                logger.error("Unexpected future error: %s", exc)

    success_count = sum(1 for r in all_results if r["status"] == "success")
    skip_count = sum(1 for r in all_results if r["status"] == "skipped")
    error_count = len(all_results) - success_count - skip_count
    _ok(
        f"Done: {success_count:,} processed | {skip_count:,} skipped | {error_count:,} errors",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 4 — Quality report
    # ------------------------------------------------------------------
    _step(4, total_steps, "Writing quality_report.json ...")
    t0 = time.perf_counter()
    q_path = OUTPUT_DIR / "quality_report.json"
    quality_writer.save(q_path)
    _ok(
        f"quality_report.json | {quality_writer.flagged}/{quality_writer.total} images flagged",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 5 — Preprocessing metadata
    # ------------------------------------------------------------------
    _step(5, total_steps, "Writing preprocessing_metadata.csv and preprocessing_report.json ...")
    t0 = time.perf_counter()
    metadata_writer.save_csv(OUTPUT_DIR / "preprocessing_metadata.csv")
    metadata_writer.save_report(OUTPUT_DIR / "preprocessing_report.json")
    _ok(
        f"preprocessing_metadata.csv: {metadata_writer.total_processed:,} rows",
        time.perf_counter() - t0,
    )

    # ------------------------------------------------------------------
    # Step 6 — Visualizations
    # ------------------------------------------------------------------
    _step(6, total_steps, "Generating before/after comparison grids ...")
    t0 = time.perf_counter()

    vis_charts: List[Path] = []
    if vis_cfg.get("enabled", True):
        samples_per_mod = int(vis_cfg.get("samples_per_modality", 5))
        success_results = [r for r in all_results if r["status"] == "success" and r["processed_path"]]
        vis_charts = generate_visualizations(success_results, OUTPUT_DIR, samples_per_mod)
    else:
        logger.info("Visualizations disabled in config.")

    _ok(f"{len(vis_charts)} comparison grid(s) saved", time.perf_counter() - t0)

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - pipeline_start
    print(f"\n{'=' * 78}")
    print("  MODULE 3 COMPLETE")
    print(f"{'=' * 78}")
    print(f"  Output directory  : {OUTPUT_DIR}")
    print(f"  Images processed  : {success_count:,}")
    print(f"  Images skipped    : {skip_count:,}")
    print(f"  Errors            : {error_count:,}")
    print(f"  Quality flagged   : {quality_writer.flagged:,}")
    print(f"  Comparison grids  : {len(vis_charts)}")
    print(f"  Total time        : {total_elapsed:.1f}s")
    print(f"{'=' * 78}\n")
    logger.info("Module 3 pipeline complete in %.1fs.", total_elapsed)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 3: Medical Image Preprocessing Framework"
    )
    parser.add_argument(
        "--sample", action="store_true",
        help="Process a small sample (sample_size per modality) for testing.",
    )
    parser.add_argument(
        "--sample-size", type=int, default=10,
        help="Number of images per modality in sample mode (default: 10).",
    )
    parser.add_argument(
        "--modality", type=str, default=None,
        help="Process only one modality (e.g., brain_mri, chest_xray).",
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
            num_workers_override=args.workers,
        )
    )
