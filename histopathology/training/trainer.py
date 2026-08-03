"""
histopathology/training/trainer.py
=====================================
HistopathologyTrainer — full training loop with AMP, cosine LR schedule,
early stopping, and checkpoint management. Mirrors Module 7 BrainMRITrainer.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint  # Module 6 reuse
from modality_classifier.training.metrics import compute_classification_metrics  # Module 6 reuse

from histopathology.utils.checkpoint_manager import CheckpointManager
from histopathology.utils.config import TrainingConfig

logger = logging.getLogger(__name__)


class HistopathologyTrainer:
    """
    Full training loop for histopathology cancer subtyping.
    Supports AMP (mixed precision), cosine LR schedule, early stopping, and resume.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        save_dir: str | Path,
        label_map: Dict[str, int],
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.label_map = label_map
        self.num_classes = len(label_map)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.epochs)
        self.scaler = torch.amp.GradScaler("cuda") if (config.use_amp and torch.cuda.is_available()) else None

        self.checkpoint_manager = CheckpointManager(save_dir)
        self.early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

        self.start_epoch = 0
        self.best_val_acc = 0.0

    def _train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            if self.scaler:
                with torch.amp.autocast("cuda"):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        metrics = compute_classification_metrics(labels_arr, probs_arr)
        metrics["loss"] = total_loss / len(self.train_loader.dataset)
        return metrics

    @torch.no_grad()
    def _val_epoch(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        all_probs, all_labels = [], []

        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        probs_arr = np.concatenate(all_probs, axis=0)
        labels_arr = np.concatenate(all_labels, axis=0)
        metrics = compute_classification_metrics(labels_arr, probs_arr)
        metrics["loss"] = total_loss / len(self.val_loader.dataset)
        return metrics

    def fit(self, resume: bool = False) -> None:
        """Run full training loop with optional resume from last checkpoint."""
        if resume and self.checkpoint_manager.has_last():
            state = self.checkpoint_manager.load_last()
            if state:
                self.model.load_state_dict(state["model_state_dict"])
                self.optimizer.load_state_dict(state["optimizer_state_dict"])
                self.start_epoch = state.get("epoch", 0) + 1
                self.best_val_acc = state.get("best_val_acc", 0.0)
                logger.info(f"Resumed training from epoch {self.start_epoch}.")

        logger.info(f"Starting histopathology training on {self.device}. Epochs: {self.config.epochs}")

        for epoch in range(self.start_epoch, self.config.epochs):
            t0 = time.time()
            train_metrics = self._train_epoch(epoch)
            val_metrics = self._val_epoch()
            self.scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch+1:03d}/{self.config.epochs}] "
                f"Train Loss={train_metrics['loss']:.4f} Acc={train_metrics['accuracy']:.4f} | "
                f"Val Loss={val_metrics['loss']:.4f} Acc={val_metrics['accuracy']:.4f} | "
                f"F1={val_metrics['f1_macro']:.4f} | Time={elapsed:.1f}s"
            )

            # Save last checkpoint for resume
            self.checkpoint_manager.save_last({
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_acc": self.best_val_acc,
            })

            # Save best model
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.checkpoint_manager.save_best({
                    "epoch": epoch,
                    "model_state_dict": self.model.state_dict(),
                    "val_accuracy": self.best_val_acc,
                    "label_map": self.label_map,
                })
                logger.info(f"  → New best model saved (Val Acc={self.best_val_acc:.4f})")

            # Early stopping
            if self.early_stopping(val_metrics["loss"]):
                logger.info(f"Early stopping triggered at epoch {epoch+1}.")
                break

        # Restore best weights
        if self.checkpoint_manager.has_best():
            state = self.checkpoint_manager.load_best()
            if state:
                self.model.load_state_dict(state["model_state_dict"])
                logger.info(f"Best model restored (Val Acc={self.best_val_acc:.4f}).")
