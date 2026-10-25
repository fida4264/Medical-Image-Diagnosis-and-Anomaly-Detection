"""
training/v2_baseline/trainer.py
===============================
Standard reproducible training loop and evaluation pipeline for V2.0 baseline models.
Handles training, validation, early stopping, checkpointing, and test evaluation.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .dataset import V2ManifestDataset
from .transforms import get_transforms
from .model_factory import create_baseline_model, V2BaselineModel
from .loss_factory import create_loss_function, compute_class_weights, compute_multilabel_pos_weights
from .metrics import calculate_metrics, calculate_multilabel_metrics

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets deterministic seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class V2Trainer:
    """
    Standard training and evaluation manager for V2.0 models.
    """

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() and config.device != "cpu" else "cpu")

        # 1. Enforce reproducibility
        set_seed(self.config.seed)

        # 2. Setup paths
        self.checkpoint_dir = Path(self.config.checkpoint_dir) / self.config.modality
        if self.config.fold is not None:
            self.checkpoint_dir = self.checkpoint_dir / f"fold_{self.config.fold}"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.output_dir = Path(self.config.output_dir) / self.config.modality
        if self.config.fold is not None:
            self.output_dir = self.output_dir / f"fold_{self.config.fold}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 3. Initialize Model
        self.model = create_baseline_model(
            model_name=self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=self.config.pretrained,
            freeze_backbone=self.config.freeze_backbone,
            task=self.config.task,
        ).to(self.device)

        # 4. Prepare Datasets & Loaders
        self.train_loader, self.val_loader, self.test_loader = self._setup_dataloaders()

        # 5. Initialize Loss function with weights if required
        self.criterion = self._setup_criterion()

        # 6. Optimizer & Scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, self.config.epochs),
            eta_min=1e-6,
        )

        self.best_metric_val = -float("inf") if self.config.early_stopping_metric != "val_loss" else float("inf")
        self.best_model_weights = None
        self.best_epoch = 0

        # Training history
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "train_loss": [],
            "val_loss": [],
            "val_macro_f1": [],
            "lr": [],
        }

    def _setup_dataloaders(self) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:
        train_transform = get_transforms(self.config.modality, is_train=True, image_size=self.config.image_size)
        eval_transform = get_transforms(self.config.modality, is_train=False, image_size=self.config.image_size)

        test_loader = None

        if self.config.split_type == "kfold":
            fold = self.config.fold if self.config.fold is not None else 0
            train_ds = V2ManifestDataset(
                manifest=self.config.split_file,
                fold=fold,
                is_train_fold=True,
                transform=train_transform,
                task=self.config.task,
                num_classes=self.config.num_classes,
            )
            val_ds = V2ManifestDataset(
                manifest=self.config.split_file,
                fold=fold,
                is_train_fold=False,
                transform=eval_transform,
                task=self.config.task,
                num_classes=self.config.num_classes,
            )
        else:
            # Fixed train / validation / test splits
            train_ds = V2ManifestDataset(
                manifest=self.config.split_file,
                split="train",
                transform=train_transform,
                task=self.config.task,
                num_classes=self.config.num_classes,
            )
            val_ds = V2ManifestDataset(
                manifest=self.config.split_file,
                split="validation",
                transform=eval_transform,
                task=self.config.task,
                num_classes=self.config.num_classes,
            )
            try:
                test_ds = V2ManifestDataset(
                    manifest=self.config.split_file,
                    split="test",
                    transform=eval_transform,
                    task=self.config.task,
                    num_classes=self.config.num_classes,
                )
                test_loader = DataLoader(
                    test_ds,
                    batch_size=self.config.batch_size,
                    shuffle=False,
                    num_workers=self.config.num_workers,
                )
            except Exception:
                test_loader = None

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
        )

        return train_loader, val_loader, test_loader

    def _setup_criterion(self) -> nn.Module:
        class_weights = None
        pos_weights = None

        if self.config.use_class_weights:
            labels = self.train_loader.dataset.get_labels()
            if self.config.task == "multilabel":
                pos_weights = compute_multilabel_pos_weights(labels)
            else:
                class_weights = compute_class_weights(labels, self.config.num_classes)

        return create_loss_function(
            loss_type=self.config.loss_type,
            task=self.config.task,
            num_classes=self.config.num_classes,
            class_weights=class_weights,
            pos_weights=pos_weights,
            device=str(self.device),
        )

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for images, targets, _ in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()

            # Gradient clipping to prevent destabilization
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    def evaluate(self, dataloader: DataLoader, is_test: bool = False) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_targets = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for images, targets, _ in dataloader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(images)
                loss = self.criterion(logits, targets)
                total_loss += loss.item()

                if self.config.task == "multilabel":
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_probs.append(probs)
                    all_targets.append(targets.cpu().numpy())
                else:
                    probs = torch.softmax(logits, dim=-1).cpu().numpy()
                    preds = np.argmax(probs, axis=-1)
                    all_probs.append(probs)
                    all_preds.append(preds)
                    all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / max(1, len(dataloader))

        if self.config.task == "multilabel":
            y_true = np.vstack(all_targets)
            y_prob = np.vstack(all_probs)
            metrics = calculate_multilabel_metrics(
                y_true=y_true,
                y_prob=y_prob,
                class_names=self.config.class_names,
            )
        else:
            y_true = np.concatenate(all_targets)
            y_pred = np.concatenate(all_preds)
            y_prob = np.vstack(all_probs)
            metrics = calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=self.config.class_names,
                modality=self.config.modality,
            )

        metrics["loss"] = avg_loss
        return metrics

    def train(self) -> Dict[str, Any]:
        """Runs the complete training and validation cycle."""
        logger.info(
            f"--- Starting Baseline Training: {self.config.modality} "
            f"(Model: {self.config.model_name}, Epochs: {self.config.epochs}) ---"
        )
        start_time = time.time()
        patience_counter = 0

        for epoch in range(1, self.config.epochs + 1):
            epoch_start = time.time()
            train_loss = self.train_epoch(epoch)
            val_metrics = self.evaluate(self.val_loader, is_test=False)
            val_loss = val_metrics["loss"]
            val_macro_f1 = val_metrics["macro_f1"]

            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            # Record history
            self.history["epoch"].append(epoch)
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_macro_f1"].append(val_macro_f1)
            self.history["lr"].append(current_lr)

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch:02d}/{self.config.epochs:02d} [{elapsed:.1f}s] - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} | LR: {current_lr:.2e}"
            )

            # Check early stopping / best model tracking
            improved = False
            if self.config.early_stopping_metric == "val_loss":
                if val_loss < self.best_metric_val:
                    self.best_metric_val = val_loss
                    improved = True
            else:
                if val_macro_f1 > self.best_metric_val:
                    self.best_metric_val = val_macro_f1
                    improved = True

            if improved:
                patience_counter = 0
                self.best_epoch = epoch
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                self._save_checkpoint(
                    filename="best_model.pt",
                    epoch=epoch,
                    metrics=val_metrics,
                    is_best=True,
                )
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch} (patience={self.config.early_stopping_patience}).")
                    break

        # Save final checkpoint
        self._save_checkpoint(
            filename="last_model.pt",
            epoch=self.history["epoch"][-1] if self.history["epoch"] else 0,
            metrics=val_metrics,
            is_best=False,
        )

        total_training_time = time.time() - start_time
        logger.info(f"Training completed in {total_training_time:.1f}s. Best Epoch: {self.best_epoch}")

        # Restore best weights for evaluation
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        # Run best validation evaluation
        best_val_metrics = self.evaluate(self.val_loader, is_test=False)

        # Strictly evaluate test set ONLY if available and applicable
        test_metrics = None
        if self.test_loader is not None and self.config.split_type == "fixed":
            logger.info("Evaluating on held-out fixed test set...")
            test_metrics = self.evaluate(self.test_loader, is_test=True)

        results = {
            "modality": self.config.modality,
            "model_name": self.config.model_name,
            "task": self.config.task,
            "best_epoch": self.best_epoch,
            "total_epochs": len(self.history["epoch"]),
            "training_time_seconds": total_training_time,
            "best_val_metrics": best_val_metrics,
            "test_metrics": test_metrics,
            "history": self.history,
            "config": self.config.to_dict(),
        }

        # Save results to output directory
        results_path = self.output_dir / "experiment_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved results to {results_path}")

        return results

    def _save_checkpoint(
        self,
        filename: str,
        epoch: int,
        metrics: Dict[str, Any],
        is_best: bool = False,
    ) -> None:
        """Saves PyTorch checkpoint with full reproducibility metadata."""
        meta = {
            "epoch": epoch,
            "metrics": metrics,
            "modality": self.config.modality,
            "model_architecture": self.config.model_name,
            "dataset_version": self.config.dataset_version,
            "split_manifest": self.config.split_file,
            "seed": self.config.seed,
            "class_names": self.config.class_names,
            "num_classes": self.config.num_classes,
            "task": self.config.task,
            "is_best": is_best,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metadata": meta,
            },
            checkpoint_path,
        )

        # Also write human-readable metadata JSON sidecar
        meta_json_path = self.checkpoint_dir / f"{Path(filename).stem}_meta.json"
        with open(meta_json_path, "w", encoding="utf-8") as f:
            # Filter non-serializable elements if any
            clean_metrics = {k: v for k, v in metrics.items() if not isinstance(v, (torch.Tensor, np.ndarray))}
            meta["metrics"] = clean_metrics
            json.dump(meta, f, indent=2)
