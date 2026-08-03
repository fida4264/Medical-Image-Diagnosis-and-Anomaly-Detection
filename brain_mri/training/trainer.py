"""
brain_mri/training/trainer.py
==============================
Training engine for Brain MRI Disease Classifier supporting Mixed Precision, Resume, and TensorBoard.
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

from brain_mri.utils.checkpoint_manager import CheckpointManager
from brain_mri.utils.config import BrainMRITrainingConfig
from modality_classifier.training.callbacks import EarlyStopping  # REUSE Module 6
from modality_classifier.training.metrics import compute_classification_metrics  # REUSE Module 6

logger = logging.getLogger(__name__)

# Optional TensorBoard support
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False


class BrainMRITrainer:
    """
    Training engine for Brain MRI Disease Model with Mixed Precision, Resume, and TensorBoard.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: BrainMRITrainingConfig,
        save_dir: str | Path = "saved_models/brain_mri",
        label_map: Optional[Dict[str, int]] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.save_dir = Path(save_dir)
        self.label_map = label_map or {}

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer & Scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Checkpoint & Early Stopping
        self.ckpt_manager = CheckpointManager(saved_model_dir=self.save_dir)
        self.early_stopping = EarlyStopping(patience=config.patience, mode="min")

        # Mixed Precision Scaler
        self.use_amp = config.mixed_precision and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        # TensorBoard
        self.writer: Optional[Any] = None
        if config.use_tensorboard and TENSORBOARD_AVAILABLE:
            tb_dir = Path(config.tensorboard_log_dir)
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging initialized at {tb_dir}")

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "val_f1": [],
            "lr": [],
        }
        self.start_epoch = 1
        self.best_val_loss = float("inf")

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
            return AdamW(self.model.parameters(), lr=lr, weight_decay=wd)

    def _build_scheduler(self) -> Optional[LRScheduler]:
        sched_name = self.config.scheduler.lower()
        if sched_name == "reduce_on_plateau":
            return ReduceLROnPlateau(self.optimizer, mode="min", factor=0.5, patience=2)
        elif sched_name == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=self.config.epochs)
        return None

    def resume_from_checkpoint(self, load_best: bool = False) -> None:
        """
        Resumes training state from checkpoint file.
        """
        epoch, val_loss, label_map = self.ckpt_manager.load_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            load_best=load_best,
            device=self.device,
        )
        if epoch > 0:
            self.start_epoch = epoch + 1
            self.best_val_loss = val_loss
            if label_map:
                self.label_map = label_map
            logger.info(f"Resuming training from epoch {self.start_epoch}")

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp and self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
                self.scaler.scale(loss).backward()
                if self.config.grad_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
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

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)
            else:
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

    def fit(self, resume: bool = False) -> Dict[str, List[float]]:
        if resume:
            self.resume_from_checkpoint()

        logger.info(
            f"Starting Brain MRI training on {self.device} (AMP={self.use_amp}) "
            f"from epoch {self.start_epoch} to {self.config.epochs}."
        )

        start_time = time.time()
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_metrics = self.validate()

            current_lr = self.optimizer.param_groups[0]["lr"]
            val_acc = val_metrics["accuracy"]
            val_f1 = val_metrics["f1_macro"]

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)
            self.history["val_f1"].append(val_f1)
            self.history["lr"].append(current_lr)

            # TensorBoard logging
            if self.writer:
                self.writer.add_scalar("Loss/Train", train_loss, epoch)
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
                self.writer.add_scalar("Accuracy/Val", val_acc, epoch)
                self.writer.add_scalar("F1/Val_Macro", val_f1, epoch)
                self.writer.add_scalar("LearningRate", current_lr, epoch)

            # LR Scheduler
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            logger.info(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] — "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val F1: {val_f1:.4f}"
            )

            self.ckpt_manager.save_checkpoint(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                epoch=epoch,
                val_loss=val_loss,
                val_f1=val_f1,
                label_map=self.label_map,
                is_best=is_best,
            )

            if self.early_stopping(val_loss):
                logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        if self.writer:
            self.writer.close()

        elapsed = time.time() - start_time
        logger.info(f"Brain MRI training completed in {elapsed:.2f} seconds.")
        return self.history
