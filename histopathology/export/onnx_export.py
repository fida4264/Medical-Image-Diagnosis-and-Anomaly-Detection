"""
histopathology/export/onnx_export.py
=======================================
ONNX exporter for the histopathology cancer subtyping model.
Mirrors Module 7/8/9 onnx_export.py pattern with dynamo=False.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class HistopathologyONNXExporter:
    """Exports the HistopathologyClassifier to ONNX format."""

    @staticmethod
    def export(
        model: nn.Module,
        label_map: Dict[str, int],
        export_path: str | Path = "saved_models/histopathology/best_model.onnx",
        opset_version: int = 17,
        input_size: tuple = (1, 3, 224, 224),
    ) -> Path:
        """Export model to ONNX using legacy TorchScript exporter (dynamo=False)."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)

        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            dynamo=False,
        )

        # Save label map
        meta_path = export_path.parent / "onnx_label_map.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"label_map": label_map, "module": "histopathology"}, f, indent=2)

        logger.info(f"ONNX model exported to '{export_path}' (opset={opset_version}).")
        return export_path
