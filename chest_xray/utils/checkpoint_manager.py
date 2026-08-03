"""
chest_xray/utils/checkpoint_manager.py
========================================
Checkpoint save/load/resume manager for Module 8. Mirrors Module 7 pattern exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Saves and loads best/last checkpoints with full training state for resume support."""

    def __init__(self, saved_model_dir: str | Path = "saved_models/chest_xray"):
        self.saved_dir = Path(saved_model_dir)
        self.saved_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.saved_dir / "best_model.pt"
        self.last_path = self.saved_dir / "last_model.pt"

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[LRScheduler],
        epoch: int,
        val_loss: float,
        val_f1: float,
        label_map: Dict[str, int],
        is_best: bool = False,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> Path:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "label_map": label_map,
            "backbone": getattr(model, "backbone_name", "efficientnet_b0"),
            "extra_metadata": extra_metadata or {},
        }

        torch.save(checkpoint, self.last_path)
        logger.info(f"Updated last checkpoint: {self.last_path} (Epoch {epoch})")

        if is_best:
            torch.save(checkpoint, self.best_path)
            logger.info(f"Updated best checkpoint: {self.best_path} (Val F1: {val_f1:.4f})")
            return self.best_path

        return self.last_path

    def load_checkpoint(
        self,
        model: nn.Module,
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[LRScheduler] = None,
        load_best: bool = True,
        device: Optional[torch.device] = None,
    ) -> Tuple[int, float, Dict[str, int]]:
        path = self.best_path if load_best and self.best_path.is_file() else self.last_path
        if not path.is_file():
            logger.warning(f"No checkpoint found at {path}. Starting fresh.")
            return 0, float("inf"), {}

        dev = device or torch.device("cpu")
        checkpoint = torch.load(path, map_location=dev)

        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        start_epoch = checkpoint.get("epoch", 0)
        val_loss = checkpoint.get("val_loss", float("inf"))
        label_map = checkpoint.get("label_map", {})

        logger.info(f"Loaded checkpoint from {path} (Epoch {start_epoch})")
        return start_epoch, val_loss, label_map
