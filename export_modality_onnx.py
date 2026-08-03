"""
export_modality_onnx.py
=======================
CLI entrypoint for exporting trained PyTorch Modality Classifier model to ONNX format.

Usage
-----
    python export_modality_onnx.py
    python export_modality_onnx.py --checkpoint outputs/models/modality_classifier_best.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from modality_classifier.config import ModalityClassifierConfig
from modality_classifier.inference.onnx_exporter import ONNXExporter
from modality_classifier.models.base_model import ModalityClassifier
from modality_classifier.utils.logging import setup_logger

logger = setup_logger("export_modality_onnx")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export PyTorch Modality Classifier to ONNX format")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained PyTorch checkpoint (.pt)")
    parser.add_argument("--output", type=str, default=None, help="Destination ONNX model path (.onnx)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ModalityClassifierConfig.load(args.config)
    checkpoint_path = Path(args.checkpoint or cfg.model.saved_model_path)
    output_path = Path(args.output or cfg.onnx.export_path)
    label_map_path = Path(cfg.data.label_map_output_path)

    # Determine num_classes
    num_classes = 7
    if label_map_path.is_file():
        with open(label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        num_classes = data.get("num_classes", 7)

    # Load model
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        backbone = checkpoint.get("metadata", {}).get("backbone", cfg.model.backbone)
        model = ModalityClassifier(num_classes=num_classes, backbone_name=backbone)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info(f"Loaded trained PyTorch checkpoint from {checkpoint_path}")
    else:
        logger.warning(f"Checkpoint file {checkpoint_path} not found. Exporting base backbone architecture.")
        model = ModalityClassifier(num_classes=num_classes, backbone_name=cfg.model.backbone)

    exporter = ONNXExporter(model=model, config=cfg.onnx)
    exported_onnx_path = exporter.export(output_path=output_path)

    # Numerical verification
    is_valid = exporter.verify(exported_onnx_path)

    if is_valid:
        logger.info(f"ONNX export and verification PASSED! Model ready for deployment at: {exported_onnx_path}")
    else:
        logger.error("ONNX verification failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
