"""
modality_classifier/training/trainer.py
========================================
PyTorch Training engine for Modality Classifier model.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD, Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, LRScheduler
from torch.utils.data import DataLoader

from modality_classifier.config import TrainingConfig
from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint
from modality_classifier.training.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class ModalityTrainer:
    """
    Manages complete model training, validation, early stopping, and checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_path: str | Path = "outputs/models/modality_classifier_best.pt",
        label_map: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_path = Path(save_path)
        self.label_map = label_map or {}

        # Set device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = self._build_optimizer()

        # Learning Rate Scheduler
        self.scheduler = self._build_scheduler()

        # Callbacks
        self.early_stopping = EarlyStopping(patience=config.patience, mode="min")
        self.checkpoint = ModelCheckpoint(filepath=self.save_path, mode="min")

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }

    def _build_optimizer(self) -> Optimizer:
        opt_name = self.config.optimizer.lower()
        lr = self.config.learning_rate
        wd = self.config.weight_decay

        if opt_name == "adamw":
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=wd)
        elif opt_name == "sgd":
            return SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
        else:
            logger.warning(f"Unknown optimizer '{opt_name}'. Defaulting to AdamW.")
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def _build_scheduler(self) -> Optional[LRScheduler]:
        sched_name = self.config.scheduler.lower()
        if sched_name == "reduce_on_plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        elif sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return None

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()
            if self.config.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(self.train_loader.dataset)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        metrics = compute_classification_metrics(targets_arr, preds_arr)

        return avg_loss, metrics["accuracy"]

    @torch.no_grad()
    def validate(self) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets in self.val_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

        avg_loss = total_loss / len(self.val_loader.dataset)
        preds_arr = np.concatenate(all_preds, axis=0)
        targets_arr = np.concatenate(all_targets, axis=0)
        metrics = compute_classification_metrics(targets_arr, preds_arr)

        return avg_loss, metrics

    def fit(self) -> Dict[str, List[float]]:
        logger.info(
            f"Starting training on {self.device} for {self.config.epochs} epochs. "
            f"Train samples: {len(self.train_loader.dataset)}, Val samples: {len(self.val_loader.dataset)}"
        )

        start_time = time.time()
        for epoch in range(1, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # Scheduler step
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            logger.info(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] — "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f} | LR: {current_lr:.6f}"
            )

            # Model Checkpoint & Early Stopping
            metadata = {
                "epoch": epoch,
                "label_map": self.label_map,
                "backbone": getattr(self.model, "backbone_name", "unknown"),
                "val_acc": val_acc,
                "val_f1": val_f1,
            }
            self.checkpoint.step(val_loss, self.model, metadata=metadata)

            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        elapsed = time.time() - start_time
        logger.info(f"Training complete in {elapsed:.2f} seconds. Best checkpoint saved to {self.save_path}")
        return self.history
