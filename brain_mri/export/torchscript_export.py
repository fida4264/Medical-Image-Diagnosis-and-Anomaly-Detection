"""
brain_mri/export/torchscript_export.py
======================================
TorchScript Exporter using static graph tracing for Brain MRI Model.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """
    Exports PyTorch model to TorchScript (.torchscript / .pt) format via torch.jit.trace.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.model.cpu()

    def export(
        self,
        output_path: str | Path = "saved_models/brain_mri/best_model.torchscript",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> Path:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_size, dtype=torch.float32)

        traced_model = torch.jit.trace(self.model, dummy_input)
        traced_model.save(str(out_path))

        logger.info(f"Successfully exported TorchScript model to {out_path}")
        return out_path
