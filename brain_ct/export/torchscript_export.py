"""
brain_ct/export/torchscript_export.py
====================================
TorchScript export functionality for Brain CT Classifier.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """Exports trained PyTorch Brain CT model to TorchScript (.torchscript)."""

    @staticmethod
    def export(
        model: nn.Module,
        export_path: str | Path = "saved_models/brain_ct/best_model.torchscript",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> Path:
        out_path = Path(export_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval().cpu()
        dummy_input = torch.randn(*input_size, device="cpu")

        scripted_model = torch.jit.trace(model, dummy_input)
        scripted_model.save(str(out_path))

        logger.info(f"Successfully exported TorchScript model to {out_path}")
        return out_path
