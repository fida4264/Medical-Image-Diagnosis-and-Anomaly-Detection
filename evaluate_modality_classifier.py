"""
evaluate_modality_classifier.py
===============================
CLI entrypoint for evaluating trained Modality Classifier model on test set.

Usage
-----
    python evaluate_modality_classifier.py
    python evaluate_modality_classifier.py --config configs/modality_classifier.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
import torch

from modality_classifier.config import ModalityClassifierConfig
from modality_classifier.datasets.data_loader import create_data_loaders
from modality_classifier.evaluation.evaluator import ModalityEvaluator
from modality_classifier.models.base_model import ModalityClassifier
from modality_classifier.utils.logging import setup_logger

logger = setup_logger("evaluate_modality_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Medical Image Modality Classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--model-path", type=str, default=None, help="Override checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModalityClassifierConfig.load(args.config)
    checkpoint_path = Path(args.model_path or cfg.model.saved_model_path)
    label_map_path = Path(cfg.data.label_map_output_path)
    splits_path = Path(cfg.data.splits_output_path)

    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint model file missing at {checkpoint_path}. Train model first!")
        sys.exit(1)

    if not label_map_path.is_file() or not splits_path.is_file():
        logger.error("Splits or label map missing. Run dataset builder or training first.")
        sys.exit(1)

    # Load label map & splits
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map_data = json.load(f)
    label_map = label_map_data["modality_to_label"]

    splits_df = pd.read_csv(splits_path)
    loaders = create_data_loaders(
        splits_df=splits_df,
        batch_size=cfg.training.batch_size,
        target_size=cfg.data.target_size,
        num_workers=0,
    )

    if "test" not in loaders:
        logger.error("Test split not found in DataLoaders.")
        sys.exit(1)

    # Load Model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    backbone = checkpoint.get("metadata", {}).get("backbone", cfg.model.backbone)

    model = ModalityClassifier(num_classes=len(label_map), backbone_name=backbone)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded model checkpoint from {checkpoint_path}")

    # Evaluate
    evaluator = ModalityEvaluator(
        model=model,
        test_loader=loaders["test"],
        label_map=label_map,
        config=cfg.evaluation,
    )

    results = evaluator.evaluate()
    evaluator.save_reports(results)

    acc = results["global_metrics"]["accuracy"]
    f1 = results["global_metrics"]["f1_macro"]
    logger.info(f"Evaluation complete! Test Accuracy: {acc*100:.2f}%, Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()
