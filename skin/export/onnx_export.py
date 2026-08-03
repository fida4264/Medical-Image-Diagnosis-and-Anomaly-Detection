"""
skin/export/onnx_export.py
===========================
ONNX exporter for the skin lesion cancer classification model.
Uses dynamo=False to avoid onnxscript dependency, matching Module 7–11 pattern.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class SkinONNXExporter:
    """Exports a trained skin lesion model to ONNX format with label map metadata."""

    @staticmethod
    def export(
        model: nn.Module,
        label_map: Dict[str, int],
        export_path: str | Path = "saved_models/skin/best_model.onnx",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 17,
    ) -> Path:
        """Export PyTorch model to ONNX."""
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval()
        device = next(model.parameters()).device
        dummy_input = torch.randn(*input_size).to(device)

        # Include dynamo=False to prevent onnxscript dependency error
        torch.onnx.export(
            model,
            dummy_input,
            str(export_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
            opset_version=opset_version,
            dynamo=False,
        )

        # Save metadata sidecar
        metadata_path = export_path.with_suffix(".json")
        inv_label_map = {v: k for k, v in label_map.items()}
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "class_to_label": label_map,
                    "label_to_class": inv_label_map,
                    "input_size": list(input_size),
                    "opset_version": opset_version,
                    "modality": "skin",
                },
                f,
                indent=2,
            )

        logger.info(f"ONNX model exported to '{export_path}' with metadata sidecar '{metadata_path}'.")
        return export_path
