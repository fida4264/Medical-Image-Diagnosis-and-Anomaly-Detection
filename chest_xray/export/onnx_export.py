"""
chest_xray/export/onnx_export.py
=================================
ONNX Export functionality for Chest X-ray Classifier.
Mirrors Module 7 ONNX exporter.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ChestXrayONNXExporter:
    """Exports trained PyTorch Chest X-ray model to ONNX format."""

    @staticmethod
    def export(
        model: nn.Module,
        label_map: Dict[str, int],
        export_path: str | Path = "saved_models/chest_xray/best_model.onnx",
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        opset_version: int = 17,
    ) -> Path:
        out_path = Path(export_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        model.eval().cpu()
        dummy_input = torch.randn(*input_size, device="cpu")

        # PyTorch 2.1+ legacy ONNX export syntax compatibility
        try:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                model,
                dummy_input,
                str(out_path),
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            )

        meta_path = out_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "onnx_model": out_path.name,
                    "input_name": "input",
                    "output_name": "output",
                    "input_shape": list(input_size),
                    "label_map": label_map,
                    "opset_version": opset_version,
                },
                f,
                indent=2,
            )

        logger.info(f"Successfully exported ONNX model to {out_path}")
        return out_path
