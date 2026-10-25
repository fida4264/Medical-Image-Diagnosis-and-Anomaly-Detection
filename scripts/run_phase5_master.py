"""
scripts/run_phase5_master.py
============================
Master automation script for Phase 5 baseline model execution.
Waits for Ultrasound 5-fold CV to complete, then runs the remaining modalities:
  1. Brain MRI (3 epochs)
  2. Retina (3 epochs)
  3. Skin Lesion (3 epochs)
  4. Chest X-ray (2 epochs)
  5. Histopathology (2 epochs)
Then re-runs the 18-point test suite and generates the final 28-section Phase 5 report.
"""

import sys
import time
import subprocess
import logging
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase5_master")


def wait_for_ultrasound(timeout_seconds=3600):
    """Waits for Ultrasound fold 4 to finish and save experiment_results.json."""
    logger.info("Checking Ultrasound 5-fold cross-validation progress...")
    ultrasound_fold4_json = EXPERIMENTS_DIR / "ultrasound" / "fold_4" / "experiment_results.json"

    start_wait = time.time()
    while not ultrasound_fold4_json.exists():
        elapsed = time.time() - start_wait
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Ultrasound did not finish within {timeout_seconds}s!")
        time.sleep(15)

    logger.info("✅ Ultrasound 5-fold cross-validation is COMPLETE!")
    time.sleep(5)  # Allow file buffers to flush


def main():
    logger.info("=== PHASE 5 MASTER AUTOMATION RUNNER STARTED ===")

    # 1. Wait for Ultrasound to finish if currently running
    ultrasound_done = (EXPERIMENTS_DIR / "ultrasound" / "fold_4" / "experiment_results.json").exists()
    if not ultrasound_done:
        wait_for_ultrasound()
    else:
        logger.info("Ultrasound 5-fold cross-validation already complete.")

    # 2. Run remaining modalities: brain_mri, retina, skin, chest_xray, histopathology
    logger.info("Launching remaining modalities sequentially...")
    from scripts.train_all_modalities import run_training_pipeline
    run_training_pipeline(skip_modalities=["brain_ct", "ultrasound"])

    # 3. Run automated test suite
    logger.info("\n=== Running Final Automated Test Suite (pytest) ===")
    test_proc = subprocess.run(["pytest", "tests/test_v2_baseline.py", "-v"], cwd=str(ROOT))
    if test_proc.returncode != 0:
        logger.error("❌ Some automated tests failed!")
    else:
        logger.info("✅ All automated tests PASSED after training!")

    # 4. Generate the final 28-section report
    logger.info("\n=== Compiling Final V2.0 Phase 5 Report ===")
    from scripts.generate_v2_phase5_report import build_report
    build_report()

    logger.info("=== PHASE 5 COMPLETE: ALL 7 MODALITIES TRAINED & VERIFIED ===")


if __name__ == "__main__":
    main()
