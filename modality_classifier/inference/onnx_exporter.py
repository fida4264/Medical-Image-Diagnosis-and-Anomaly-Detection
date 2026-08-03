"""
modality_classifier/inference/onnx_exporter.py
===============================================
PyTorch to ONNX exporter and numerical verification tool.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from modality_classifier.config import OnnxConfig
from modality_classifier.models.base_model import ModalityClassifier

logger = logging.getLogger(__name__)


class ONNXExporter:
    """
    Exports trained PyTorch ModalityClassifier model to ONNX format and verifies accuracy.
    """

    def __init__(self, model: nn.Module, config: Optional[OnnxConfig] = None):
        self.model = model
        self.config = config or OnnxConfig()
        self.model.eval()
        self.model.cpu()

    def export(
        self,
        output_path: Optional[str | Path] = None,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> Path:
        """
        Exports PyTorch model to ONNX.
        """
        out_path = Path(output_path or self.config.export_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_size, dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=[self.config.input_name],
            output_names=[self.config.output_name],
            dynamic_axes=self.config.dynamic_axes,
            dynamo=False,  # Use legacy TorchScript exporter — does NOT require onnxscript
        )

        logger.info(f"Successfully exported ONNX model to {out_path}")
        return out_path

    def verify(
        self,
        onnx_path: str | Path,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        tolerance: float = 1e-4,
    ) -> bool:
        """
        Verifies exported ONNX model against PyTorch predictions using ONNXRuntime.
        """
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX model file not found: {path}")

        # Check ONNX structure
        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

        # PyTorch prediction
        dummy_input_np = np.random.randn(*input_size).astype(np.float32)
        dummy_input_torch = torch.from_numpy(dummy_input_np)

        with torch.no_grad():
            pytorch_logits = self.model(dummy_input_torch).numpy()

        # ONNXRuntime prediction
        ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_inputs = {self.config.input_name: dummy_input_np}
        ort_outputs = ort_session.run(None, ort_inputs)
        onnx_logits = ort_outputs[0]

        # Calculate max absolute difference
        max_diff = float(np.max(np.abs(pytorch_logits - onnx_logits)))
        is_valid = max_diff < tolerance

        if is_valid:
            logger.info(f"ONNX numerical verification PASSED! Max abs difference: {max_diff:.6e} < {tolerance}")
        else:
            logger.error(f"ONNX numerical verification FAILED! Max abs difference: {max_diff:.6e} >= {tolerance}")

        return is_valid
