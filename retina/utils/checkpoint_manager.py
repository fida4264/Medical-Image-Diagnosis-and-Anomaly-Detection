"""
retina/utils/checkpoint_manager.py
====================================
Checkpoint save/load/resume manager for Module 11. Mirrors Module 7 pattern exactly.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving and loading model checkpoints for the retina module."""

    BEST_FILENAME = "best_model.pt"
    LAST_FILENAME = "last_model.pt"

    def __init__(self, save_dir: str | Path = "saved_models/retina"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    @property
    def best_path(self) -> Path:
        return self.save_dir / self.BEST_FILENAME

    @property
    def last_path(self) -> Path:
        return self.save_dir / self.LAST_FILENAME

    def save_best(self, state: Dict[str, Any]) -> None:
        """Save the best model checkpoint."""
        torch.save(state, self.best_path)
        logger.info(f"Best checkpoint saved to '{self.best_path}'.")

    def save_last(self, state: Dict[str, Any]) -> None:
        """Save the last epoch checkpoint (for resume)."""
        torch.save(state, self.last_path)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best model checkpoint if it exists."""
        if self.best_path.is_file():
            state = torch.load(self.best_path, map_location="cpu", weights_only=False)
            logger.info(f"Best checkpoint loaded from '{self.best_path}'.")
            return state
        logger.warning(f"No best checkpoint found at '{self.best_path}'.")
        return None

    def load_last(self) -> Optional[Dict[str, Any]]:
        """Load the last epoch checkpoint for training resume."""
        if self.last_path.is_file():
            state = torch.load(self.last_path, map_location="cpu", weights_only=False)
            logger.info(f"Last checkpoint loaded from '{self.last_path}' for resume.")
            return state
        return None

    def has_best(self) -> bool:
        return self.best_path.is_file()

    def has_last(self) -> bool:
        return self.last_path.is_file()
