"""
ultrasound/export/torchscript_export.py
========================================
TorchScript exporter for the ultrasound disease classification model.
Mirrors Module 7–12 torchscript_export.py pattern.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class TorchScriptExporter:
    """Exports an ultrasound model to TorchScript (trace-based)."""

    @staticmethod
    def export(
        model: nn.Module,
        export_path: str | Path = "saved_models/ultrasound/best_model.torchscript",
        input_size: tuple = (1, 3, 224, 224),
    ) -> Path:
        """Trace and save the model as a TorchScript file."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)

        with torch.no_grad():
            traced = torch.jit.trace(model, dummy_input)

        traced.save(str(export_path))
        logger.info(f"TorchScript model exported to '{export_path}'.")
        return export_path
