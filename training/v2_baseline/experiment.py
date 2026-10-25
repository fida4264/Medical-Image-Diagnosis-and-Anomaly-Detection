"""
training/v2_baseline/experiment.py
==================================
Experiment registry for tracking, logging, and aggregating V2.0 baseline model results.
Maintains outputs/v2_baseline_results.csv and outputs/experiments/ artifacts.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = ROOT / "outputs"
EXPERIMENTS_DIR = OUTPUTS_DIR / "experiments"
CSV_RESULTS_PATH = OUTPUTS_DIR / "v2_baseline_results.csv"

CSV_HEADERS = [
    "modality",
    "model",
    "dataset_version",
    "seed",
    "split_strategy",
    "fold",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_auc",
    "loss",
    "notes",
]


class ExperimentRegistry:
    """Manages experiment records and CSV result aggregation."""

    def __init__(self, output_csv: Path | str = CSV_RESULTS_PATH):
        self.output_csv = Path(output_csv)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_csv_headers()

    def _ensure_csv_headers(self) -> None:
        if not self.output_csv.exists():
            with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADERS)

    def log_result(
        self,
        modality: str,
        model: str,
        dataset_version: str,
        seed: int,
        split_strategy: str,
        fold: Optional[int | str],
        metrics: Dict[str, Any],
        loss: float,
        notes: str = "",
    ) -> None:
        """Appends a new result record to the aggregate results CSV."""
        accuracy = metrics.get("accuracy", "NA")
        macro_f1 = metrics.get("macro_f1", "NA")
        weighted_f1 = metrics.get("weighted_f1", "NA")
        macro_auc = metrics.get("macro_auc", "NA")

        row = [
            modality,
            model,
            dataset_version,
            seed,
            split_strategy,
            fold if fold is not None else "NA",
            f"{accuracy:.4f}" if isinstance(accuracy, (int, float)) else "NA",
            f"{macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else "NA",
            f"{weighted_f1:.4f}" if isinstance(weighted_f1, (int, float)) else "NA",
            f"{macro_auc:.4f}" if isinstance(macro_auc, (int, float)) else "NA",
            f"{loss:.4f}" if isinstance(loss, (int, float)) else "NA",
            notes,
        ]

        with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(row)
        logger.info(f"Appended experiment result for {modality} to {self.output_csv}")

    def save_modality_artifacts(
        self,
        modality: str,
        results: Dict[str, Any],
        fold: Optional[int] = None,
    ) -> Path:
        """Saves detailed experiment artifacts (JSON metrics, confusion matrix) in outputs/experiments/."""
        modality_dir = EXPERIMENTS_DIR / modality
        if fold is not None:
            modality_dir = modality_dir / f"fold_{fold}"
        modality_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save full results JSON
        json_file = modality_dir / "experiment_results.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # 2. Save confusion matrix CSV if present
        best_val = results.get("best_val_metrics", {})
        if "confusion_matrix" in best_val:
            cm = best_val["confusion_matrix"]
            cm_df = pd.DataFrame(cm)
            cm_df.to_csv(modality_dir / "confusion_matrix.csv", index=False)

        # 3. Save training history CSV
        history = results.get("history", {})
        if history and "epoch" in history:
            hist_df = pd.DataFrame(history)
            hist_df.to_csv(modality_dir / "training_history.csv", index=False)

        logger.info(f"Saved modality experiment artifacts to {modality_dir}")
        return modality_dir
