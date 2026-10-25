"""
run_v2_baseline.py
==================
Unified CLI entrypoint for training and evaluating V2.0 baseline medical image models.

Usage examples:
  # Smoke test on all modalities
  python run_v2_baseline.py --smoke_test

  # Train Brain MRI baseline
  python run_v2_baseline.py --modality brain_mri --epochs 10

  # Train Brain CT all 5 folds
  python run_v2_baseline.py --modality brain_ct --fold all --epochs 10

  # Train all modalities with baseline configurations
  python run_v2_baseline.py --modality all
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch

from training.v2_baseline.config import ExperimentConfig, MODALITY_METADATA
from training.v2_baseline.trainer import V2Trainer
from training.v2_baseline.experiment import ExperimentRegistry
from training.v2_baseline.smoke_test import run_all_smoke_tests, run_modality_smoke_test

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_v2_baseline")


def train_modality_folds(
    modality: str,
    model_name: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    registry: ExperimentRegistry = None,
    device: str = "cpu",
):
    """Executes all 5 folds for cross-validation modalities (Brain CT & Ultrasound)."""
    meta = MODALITY_METADATA[modality]
    k_folds = meta.get("k_folds", 5)
    logger.info(f"=== Running 5-Fold Cross-Validation for {modality} ({k_folds} folds) ===")

    fold_metrics = []

    for fold in range(k_folds):
        logger.info(f"\n--- {modality.upper()} | FOLD {fold}/{k_folds - 1} ---")
        config = ExperimentConfig.from_modality(
            modality=modality,
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=lr,
            fold=fold,
            device=device,
        )
        trainer = V2Trainer(config)
        results = trainer.train()

        val_metrics = results["best_val_metrics"]
        fold_metrics.append(val_metrics)

        if registry:
            registry.log_result(
                modality=modality,
                model=model_name,
                dataset_version=config.dataset_version,
                seed=config.seed,
                split_strategy=f"stratified_5fold_cv",
                fold=fold,
                metrics=val_metrics,
                loss=val_metrics.get("loss", 0.0),
                notes=f"Fold {fold} of 5-fold CV",
            )
            registry.save_modality_artifacts(modality, results, fold=fold)

    # Compute mean ± standard deviation across folds
    accuracies = [m["accuracy"] for m in fold_metrics if "accuracy" in m]
    macro_f1s = [m["macro_f1"] for m in fold_metrics if "macro_f1" in m]
    weighted_f1s = [m["weighted_f1"] for m in fold_metrics if "weighted_f1" in m]
    losses = [m["loss"] for m in fold_metrics if "loss" in m]

    mean_acc = float(np.mean(accuracies)) if accuracies else 0.0
    std_acc = float(np.std(accuracies)) if accuracies else 0.0
    mean_f1 = float(np.mean(macro_f1s)) if macro_f1s else 0.0
    std_f1 = float(np.std(macro_f1s)) if macro_f1s else 0.0
    mean_loss = float(np.mean(losses)) if losses else 0.0

    logger.info(f"\n=== {modality.upper()} 5-FOLD CV SUMMARY ===")
    logger.info(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    logger.info(f"Mean Macro-F1: {mean_f1:.4f} ± {std_f1:.4f}")

    if registry:
        cv_metrics = {
            "accuracy": mean_acc,
            "macro_f1": mean_f1,
            "weighted_f1": float(np.mean(weighted_f1s)) if weighted_f1s else 0.0,
            "macro_auc": float(np.mean([m["macro_auc"] for m in fold_metrics if m.get("macro_auc") is not None])) if any(m.get("macro_auc") is not None for m in fold_metrics) else None,
        }
        registry.log_result(
            modality=modality,
            model=model_name,
            dataset_version="2.0.0",
            seed=42,
            split_strategy="stratified_5fold_cv",
            fold="MEAN±STD",
            metrics=cv_metrics,
            loss=mean_loss,
            notes=f"CV 5-fold mean: Acc={mean_acc:.4f}±{std_acc:.4f}, F1={mean_f1:.4f}±{std_f1:.4f}",
        )

    return fold_metrics


def train_modality_single(
    modality: str,
    model_name: str = "resnet18",
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    registry: ExperimentRegistry = None,
    device: str = "cpu",
):
    """Executes training and evaluation on fixed train/val/test splits."""
    logger.info(f"\n=== Running Baseline Training for {modality} ===")
    config = ExperimentConfig.from_modality(
        modality=modality,
        model_name=model_name,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        device=device,
    )
    trainer = V2Trainer(config)
    results = trainer.train()

    val_metrics = results["best_val_metrics"]
    test_metrics = results.get("test_metrics")

    if registry:
        registry.log_result(
            modality=modality,
            model=model_name,
            dataset_version=config.dataset_version,
            seed=config.seed,
            split_strategy="fixed_validation",
            fold="val",
            metrics=val_metrics,
            loss=val_metrics.get("loss", 0.0),
            notes="Validation split evaluation",
        )
        if test_metrics:
            registry.log_result(
                modality=modality,
                model=model_name,
                dataset_version=config.dataset_version,
                seed=config.seed,
                split_strategy="fixed_test",
                fold="test",
                metrics=test_metrics,
                loss=test_metrics.get("loss", 0.0),
                notes="Held-out test split evaluation (unaltered test partition)",
            )
        registry.save_modality_artifacts(modality, results)

    return results


def main():
    parser = argparse.ArgumentParser(description="V2.0 Baseline Model Training & Experiment Framework")
    parser.add_argument(
        "--modality",
        type=str,
        default="smoke",
        choices=list(MODALITY_METADATA.keys()) + ["all", "smoke"],
        help="Modality to train or 'all'",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "efficientnet_b0"],
        help="Transfer learning backbone architecture",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs override")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--fold", type=str, default=None, help="Fold index (0..4) or 'all'")
    parser.add_argument("--smoke_test", action="store_true", help="Run rapid smoke test")
    parser.add_argument("--device", type=str, default="cpu", help="Device ('cpu' or 'cuda')")

    args = parser.parse_args()

    if args.smoke_test or args.modality == "smoke":
        logger.info("Executing smoke test across all 7 modalities...")
        report = run_all_smoke_tests()
        if report["all_passed"]:
            logger.info("All 7 modality smoke tests passed successfully!")
            sys.exit(0)
        else:
            logger.error("Some smoke tests failed.")
            sys.exit(1)

    registry = ExperimentRegistry()

    modalities_to_run: List[str] = list(MODALITY_METADATA.keys()) if args.modality == "all" else [args.modality]

    for mod in modalities_to_run:
        meta = MODALITY_METADATA[mod]
        epochs = args.epochs if args.epochs is not None else meta["recommended_epochs"]

        if meta["split_type"] == "kfold":
            if args.fold == "all" or args.fold is None:
                train_modality_folds(
                    modality=mod,
                    model_name=args.model,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    registry=registry,
                    device=args.device,
                )
            else:
                fold_num = int(args.fold)
                config = ExperimentConfig.from_modality(
                    modality=mod,
                    model_name=args.model,
                    epochs=epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.lr,
                    fold=fold_num,
                    device=args.device,
                )
                trainer = V2Trainer(config)
                res = trainer.train()
                registry.log_result(
                    modality=mod,
                    model=args.model,
                    dataset_version=config.dataset_version,
                    seed=config.seed,
                    split_strategy=f"fold_{fold_num}",
                    fold=fold_num,
                    metrics=res["best_val_metrics"],
                    loss=res["best_val_metrics"].get("loss", 0.0),
                    notes=f"Single fold {fold_num}",
                )
                registry.save_modality_artifacts(mod, res, fold=fold_num)
        else:
            train_modality_single(
                modality=mod,
                model_name=args.model,
                epochs=epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                registry=registry,
                device=args.device,
            )

    logger.info("\n=== Phase 5 Baseline Run Finished! Results aggregated in outputs/v2_baseline_results.csv ===")


if __name__ == "__main__":
    main()
