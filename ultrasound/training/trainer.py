"""
ultrasound/training/trainer.py
==============================
Trainer for Module 13: Ultrasound Disease Analysis Framework.
Handles training loop, AMP, early stopping, LR scheduling, checkpointing, TensorBoard logging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint  # Module 6 reuse
from modality_classifier.training.metrics import compute_classification_metrics  # Module 6 reuse
from ultrasound.utils.checkpoint_manager import CheckpointManager
from ultrasound.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


class UltrasoundTrainer:
    """Trainer engine for ultrasound disease classification."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_dir: str | Path = "saved_models/ultrasound",
        label_map: Optional[Dict[str, int]] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.label_map = label_map or {}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs,
            eta_min=1e-6,
        )

        self.scaler = torch.amp.GradScaler('cuda', enabled=config.use_amp and self.device.type == "cuda")
        self.checkpoint_manager = CheckpointManager(save_dir=self.save_dir)

        self.early_stopping = EarlyStopping(
            patience=config.patience,
            min_delta=config.min_delta,
            mode="max",
        )

        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    def train_epoch(self) -> float:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        n_samples = 0

        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            bs = images.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

        return total_loss / max(n_samples, 1)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate model and return evaluation metrics."""
        self.model.eval()
        total_loss = 0.0
        n_samples = 0
        all_targets = []
        all_probs = []

        for images, labels in self.val_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast('cuda', enabled=self.config.use_amp and self.device.type == "cuda"):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            bs = images.size(0)
            total_loss += loss.item() * bs
            n_samples += bs

            all_targets.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        val_loss = total_loss / max(n_samples, 1)
        y_true = np.array(all_targets)
        y_probs = np.array(all_probs)

        # Module 6 compute_classification_metrics API
        metrics = compute_classification_metrics(y_true, y_probs)
        metrics["loss"] = val_loss
        return metrics

    def fit(self, resume: bool = False) -> Dict[str, list]:
        """Run full training pipeline."""
        start_epoch = 0
        best_val_f1 = -1.0

        if resume and self.checkpoint_manager.has_last():
            state = self.checkpoint_manager.load_last()
            if state:
                self.model.load_state_dict(state["model_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                start_epoch = state.get("epoch", 0) + 1
                best_val_f1 = state.get("best_val_f1", -1.0)
                logger.info(f"Resumed training from epoch {start_epoch}.")

        logger.info(f"Starting ultrasound model training for {self.config.epochs} epochs...")

        for epoch in range(start_epoch, self.config.epochs):
            train_loss = self.train_epoch()
            val_metrics = self.validate()
            self.scheduler.step()

            val_loss = val_metrics["loss"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)

            logger.info(
                f"Epoch [{epoch+1}/{self.config.epochs}] - "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
            )

            state = {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_metrics": val_metrics,
                "val_accuracy": val_acc,
                "val_f1": val_f1,
                "best_val_f1": max(best_val_f1, val_f1),
                "label_map": self.label_map,
            }

            self.checkpoint_manager.save_last(state)

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                self.checkpoint_manager.save_best(state)

            if self.early_stopping(val_f1):
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        return self.history
