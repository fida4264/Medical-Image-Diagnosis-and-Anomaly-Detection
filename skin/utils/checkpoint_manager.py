"""
skin/utils/checkpoint_manager.py
===================================
Checkpoint manager for Module 12: Skin Lesion Cancer Classification Framework.
Handles saving/loading best_model.pt and last_model.pt.
Mirrors Module 7–11 pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages saving, loading, and resuming PyTorch model checkpoints."""

    def __init__(self, save_dir: str | Path = "saved_models/skin"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.best_path = self.save_dir / "best_model.pt"
        self.last_path = self.save_dir / "last_model.pt"

    def save_best(self, state_dict: Dict[str, Any]) -> None:
        """Save state_dict as best_model.pt."""
        torch.save(state_dict, self.best_path)
        logger.info(f"Saved new best model checkpoint to '{self.best_path}'.")

    def save_last(self, state_dict: Dict[str, Any]) -> None:
        """Save state_dict as last_model.pt."""
        torch.save(state_dict, self.last_path)

    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load best_model.pt if it exists."""
        if self.best_path.is_file():
            return torch.load(self.best_path, map_location="cpu", weights_only=False)
        return None

    def load_last(self) -> Optional[Dict[str, Any]]:
        """Load last_model.pt if it exists."""
        if self.last_path.is_file():
            return torch.load(self.last_path, map_location="cpu", weights_only=False)
        return None

    def has_best(self) -> bool:
        return self.best_path.is_file()

    def has_last(self) -> bool:
        return self.last_path.is_file()
