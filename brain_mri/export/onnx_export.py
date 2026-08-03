"""
brain_mri/export/onnx_export.py
===============================
ONNX Exporter and numerical verifier for Brain MRI model.
Reuses ONNXExporter pattern from Module 6 with dynamo=False. Zero external onnxscript dependency.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

from brain_mri.utils.config import BrainMRIExportConfig

logger = logging.getLogger(__name__)


class BrainMRIONNXExporter:
    """
    Exports BrainMRIClassifier to ONNX and runs verification.
    """

    def __init__(self, model: nn.Module, config: Optional[BrainMRIExportConfig] = None):
        self.model = model
        self.config = config or BrainMRIExportConfig()
        self.model.eval()
        self.model.cpu()

    def export(
        self,
        output_path: Optional[str | Path] = None,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
    ) -> Path:
        out_path = Path(output_path or self.config.onnx_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy_input = torch.randn(*input_size, dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy_input,
            str(out_path),
            export_params=True,
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            input_names=["brain_mri_input"],
            output_names=["disease_logits"],
            dynamic_axes={
                "brain_mri_input": {0: "batch_size"},
                "disease_logits": {0: "batch_size"},
            },
            dynamo=False,  # Legacy TorchScript-based exporter
        )

        logger.info(f"Successfully exported Brain MRI ONNX model to {out_path}")
        return out_path

    def verify(
        self,
        onnx_path: str | Path,
        input_size: Tuple[int, int, int, int] = (1, 3, 224, 224),
        tolerance: float = 1e-4,
    ) -> bool:
        path = Path(onnx_path)
        if not path.is_file():
            raise FileNotFoundError(f"ONNX file not found: {path}")

        onnx_model = onnx.load(str(path))
        onnx.checker.check_model(onnx_model)

        dummy_np = np.random.randn(*input_size).astype(np.float32)
        dummy_torch = torch.from_numpy(dummy_np)

        with torch.no_grad():
            pytorch_logits = self.model(dummy_torch).numpy()

        ort_session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        ort_outputs = ort_session.run(None, {"brain_mri_input": dummy_np})
        onnx_logits = ort_outputs[0]

        max_diff = float(np.max(np.abs(pytorch_logits - onnx_logits)))
        is_valid = max_diff < tolerance

        if is_valid:
            logger.info(f"Brain MRI ONNX verification PASSED! Max diff: {max_diff:.6e} < {tolerance}")
        else:
            logger.error(f"Brain MRI ONNX verification FAILED! Max diff: {max_diff:.6e} >= {tolerance}")

        return is_valid
