"""
modality_classifier/training/callbacks.py
=========================================
Training callbacks: Early stopping, model checkpointing, and logger.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to stop training when validation loss stops improving.
    """

    def __init__(self, patience: int = 3, min_delta: float = 1e-4, mode: str = "min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, current_score: float) -> bool:
        if self.best_score is None:
            self.best_score = current_score
            return False

        improved = (
            (current_score < self.best_score - self.min_delta)
            if self.mode == "min"
            else (current_score > self.best_score + self.min_delta)
        )

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class ModelCheckpoint:
    """
    Saves PyTorch model weights when validation score improves.
    """

    def __init__(self, filepath: str | Path, mode: str = "min"):
        self.filepath = Path(filepath)
        self.mode = mode
        self.best_score: Optional[float] = None
        self.filepath.parent.mkdir(parents=True, exist_ok=True)

    def step(self, current_score: float, model: nn.Module, metadata: Optional[dict] = None) -> bool:
        improved = False
        if self.best_score is None:
            improved = True
        elif self.mode == "min" and current_score < self.best_score:
            improved = True
        elif self.mode == "max" and current_score > self.best_score:
            improved = True

        if improved:
            self.best_score = current_score
            save_dict = {
                "model_state_dict": model.state_dict(),
                "best_score": current_score,
                "metadata": metadata or {},
            }
            torch.save(save_dict, self.filepath)
            logger.info(f"Saved best model checkpoint to {self.filepath} (score: {current_score:.4f})")

        return improved
