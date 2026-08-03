"""
train_modality_classifier.py
=============================
CLI entrypoint for training the Medical Image Modality Classification model.

Usage
-----
    python train_modality_classifier.py                           # Full dataset training
    python train_modality_classifier.py --sample --sample-size 50 # Quick sample test run
    python train_modality_classifier.py --epochs 5 --batch-size 64
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from modality_classifier.config import ModalityClassifierConfig
from modality_classifier.datasets.data_loader import create_data_loaders
from modality_classifier.datasets.dataset_builder import ModalityDatasetBuilder
from modality_classifier.models.base_model import build_modality_classifier
from modality_classifier.training.trainer import ModalityTrainer
from modality_classifier.utils.logging import setup_logger
from modality_classifier.utils.visualization import plot_training_curves

logger = setup_logger("train_modality_classifier")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Medical Image Modality Classifier")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone architecture")
    parser.add_argument("--sample", action="store_true", help="Run sample training mode for fast dry-run")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size per modality in sample mode")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load configuration
    cfg = ModalityClassifierConfig.load(args.config)
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.batch_size:
        cfg.training.batch_size = args.batch_size
    if args.lr:
        cfg.training.learning_rate = args.lr
    if args.backbone:
        cfg.model.backbone = args.backbone
    if args.num_workers:
        cfg.data.num_workers = args.num_workers

    logger.info("Initializing Modality Dataset Builder...")
    builder = ModalityDatasetBuilder(
        metadata_path=cfg.data.metadata_path,
        processed_dir=cfg.data.processed_images_dir,
        splits_output_path=cfg.data.splits_output_path,
        label_map_output_path=cfg.data.label_map_output_path,
        split_ratios=cfg.data.split_ratios,
        random_seed=cfg.data.random_seed,
    )

    builder.load_and_prepare(sample_mode=args.sample, sample_size=args.sample_size)
    splits_df = builder.create_splits()

    num_classes = len(builder.label_map)
    logger.info(f"Loaded {num_classes} target modalities: {builder.label_map}")

    # Build DataLoaders
    loaders = create_data_loaders(
        splits_df=splits_df,
        batch_size=cfg.training.batch_size,
        target_size=cfg.data.target_size,
        num_workers=cfg.data.num_workers,
    )

    if "train" not in loaders or "val" not in loaders:
        logger.error("Failed to build train or validation DataLoader.")
        sys.exit(1)

    # Build Model
    model = build_modality_classifier(cfg.model, num_classes=num_classes)
    logger.info(f"Built ModalityClassifier model with backbone: '{cfg.model.backbone}'")

    # Instantiate Trainer
    trainer = ModalityTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg.training,
        save_path=cfg.model.saved_model_path,
        label_map=builder.label_map,
    )

    # Execute training loop
    history = trainer.fit()

    # Plot training curves
    plot_training_curves(history, save_path="outputs/modality_training_curves.png")
    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
