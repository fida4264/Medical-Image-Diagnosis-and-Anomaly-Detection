"""
run_brain_mri.py
================
Main CLI entrypoint for Module 7: Brain MRI Disease Analysis Framework.

Usage
-----
    python run_brain_mri.py                           # Full pipeline execution
    python run_brain_mri.py --sample --sample-size 50 # Quick sample dry-run
    python run_brain_mri.py --resume                  # Resume training from last checkpoint
    python run_brain_mri.py --eval-only               # Evaluate existing checkpoint
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from brain_mri.datasets.dataset_builder import create_brain_mri_data_loaders
from brain_mri.datasets.dataset_loader import BrainMRIDatasetLoader
from brain_mri.evaluation.evaluator import BrainMRIEvaluator
from brain_mri.export.onnx_export import BrainMRIONNXExporter
from brain_mri.export.torchscript_export import TorchScriptExporter
from brain_mri.models.brain_mri_model import build_brain_mri_model
from brain_mri.training.trainer import BrainMRITrainer
from brain_mri.utils.config import BrainMRIConfig
from brain_mri.utils.logger import get_brain_mri_logger

logger = get_brain_mri_logger("run_brain_mri")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Brain MRI Disease Analysis Framework (Module 7)")
    parser.add_argument("--config", type=str, default=None, help="Path to brain_mri.yaml config")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone architecture")
    parser.add_argument("--sample", action="store_true", help="Sample mode for quick dry run")
    parser.add_argument("--sample-size", type=int, default=50, help="Sample size per class in sample mode")
    parser.add_argument("--resume", action="store_true", help="Resume training from last checkpoint")
    parser.add_argument("--eval-only", action="store_true", help="Run evaluation only on existing checkpoint")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader num_workers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = BrainMRIConfig.load(args.config)
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

    logger.info("==================================================================")
    logger.info("       Module 7: Brain MRI Disease Analysis Framework             ")
    logger.info("==================================================================")

    # 1. Dataset Loading & Dynamic Class Discovery
    loader = BrainMRIDatasetLoader(
        metadata_path=cfg.data.metadata_path,
        processed_dir=cfg.data.processed_images_dir,
        splits_output_path=cfg.data.splits_output_path,
        label_map_output_path=cfg.data.label_map_output_path,
        split_ratios=cfg.data.split_ratios,
        random_seed=cfg.data.random_seed,
    )

    loader.load_and_filter(sample_mode=args.sample, sample_size=args.sample_size)
    splits_df = loader.prepare_splits()

    num_classes = len(loader.label_map)
    logger.info(f"Loaded {num_classes} Brain MRI disease classes: {loader.label_map}")

    # 2. Build DataLoaders
    loaders = create_brain_mri_data_loaders(
        splits_df=splits_df,
        batch_size=cfg.training.batch_size,
        target_size=cfg.data.target_size,
        num_workers=cfg.data.num_workers,
    )

    if "train" not in loaders or "val" not in loaders or "test" not in loaders:
        logger.error("Failed to build train/val/test DataLoaders.")
        sys.exit(1)

    # 3. Build Model
    model = build_brain_mri_model(cfg.model, num_classes=num_classes)
    logger.info(f"Built BrainMRIClassifier model with backbone: '{cfg.model.backbone}'")

    # 4. Training (unless --eval-only)
    if not args.eval_only:
        trainer = BrainMRITrainer(
            model=model,
            train_loader=loaders["train"],
            val_loader=loaders["val"],
            config=cfg.training,
            save_dir=cfg.model.saved_model_dir,
            label_map=loader.label_map,
        )

        _ = trainer.fit(resume=args.resume)

    # 5. Evaluation
    logger.info("Starting test evaluation...")
    evaluator = BrainMRIEvaluator(
        model=model,
        test_loader=loaders["test"],
        label_map=loader.label_map,
        config=cfg.evaluation,
    )

    eval_results = evaluator.evaluate()
    evaluator.save_reports(eval_results)

    # 6. Export Models (ONNX + TorchScript)
    logger.info("Exporting production models...")
    onnx_exporter = BrainMRIONNXExporter(model=model, config=cfg.export)
    onnx_path = onnx_exporter.export(output_path=cfg.export.onnx_path)
    onnx_exporter.verify(onnx_path)

    ts_exporter = TorchScriptExporter(model=model)
    ts_exporter.export(output_path=cfg.export.torchscript_path)

    acc = eval_results["global_metrics"]["accuracy"]
    f1 = eval_results["global_metrics"]["f1_macro"]
    logger.info(f"Module 7 execution complete! Test Accuracy: {acc*100:.2f}%, Macro F1: {f1:.4f}")


if __name__ == "__main__":
    main()
