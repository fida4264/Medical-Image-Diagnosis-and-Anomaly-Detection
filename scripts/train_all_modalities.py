"""
scripts/train_all_modalities.py
===============================
Automates baseline model training sequentially across all 7 medical imaging modalities.
Logs all results to outputs/v2_baseline_results.csv and saves checkpoints to models/v2_baseline/.
Finally calls scripts/generate_v2_phase5_report.py to generate the complete Phase 5 report.
"""

import sys
import time
import logging
from pathlib import Path

from run_v2_baseline import (
    train_modality_folds,
    train_modality_single,
    ExperimentRegistry,
    MODALITY_METADATA,
)
from scripts.generate_v2_phase5_report import build_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_all_modalities")

# Balanced epoch budget for reliable baseline numbers within CPU compute limits
EPOCH_BUDGET = {
    "brain_ct": 5,        # 5 folds (200 images total)
    "ultrasound": 3,      # 5 folds (780 images total)
    "brain_mri": 3,       # 5,040 train / 560 val / 1,600 test
    "retina": 3,          # 5,119 train / 636 val / 637 test
    "skin": 3,            # 7,984 train / 1,005 val / 1,026 test
    "chest_xray": 2,      # 16,931 train / 2,117 val / 2,117 test
    "histopathology": 2,  # 20,000 train / 2,500 val / 2,500 test
}


def run_training_pipeline(skip_modalities=None):
    if skip_modalities is None:
        skip_modalities = []

    registry = ExperimentRegistry()
    start_total = time.time()

    modalities = [
        "brain_ct",
        "ultrasound",
        "brain_mri",
        "retina",
        "skin",
        "chest_xray",
        "histopathology",
    ]

    for mod in modalities:
        if mod in skip_modalities:
            logger.info(f"Skipping already trained modality: {mod}")
            continue

        meta = MODALITY_METADATA[mod]
        epochs = EPOCH_BUDGET.get(mod, 3)

        logger.info(f"\n=======================================================")
        logger.info(f"   STARTING BASELINE TRAINING: {mod.upper()} ({epochs} epochs)")
        logger.info(f"=======================================================")

        if meta["split_type"] == "kfold":
            train_modality_folds(
                modality=mod,
                model_name="resnet18",
                epochs=epochs,
                batch_size=32,
                lr=1e-4,
                registry=registry,
                device="cpu",
            )
        else:
            train_modality_single(
                modality=mod,
                model_name="resnet18",
                epochs=epochs,
                batch_size=32,
                lr=1e-4,
                registry=registry,
                device="cpu",
            )

    total_time = time.time() - start_total
    logger.info(f"\n=== ALL MODALITY BASELINES COMPLETED IN {total_time/60:.1f} MINUTES ===")

    # Compile final report
    build_report()


if __name__ == "__main__":
    skip = sys.argv[1:] if len(sys.argv) > 1 else []
    run_training_pipeline(skip_modalities=skip)
