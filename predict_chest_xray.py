"""
predict_chest_xray.py
======================
CLI single-image inference tool for Module 8: Chest X-ray Disease Analysis Framework.
Output format: Standardized JSON output and Clinical JSON report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from chest_xray.inference.predict_chest_xray import ChestXrayPredictor
from chest_xray.models.chest_xray_model import build_chest_xray_model
from chest_xray.reports.clinical_report import ClinicalReportGenerator
from chest_xray.utils.checkpoint_manager import CheckpointManager
from chest_xray.utils.config import ChestXrayConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict disease from a Chest X-ray image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input Chest X-ray image")
    parser.add_argument("--config", type=str, default=None, help="Path to chest_xray.yaml")
    parser.add_argument("--report", action="store_true", help="Generate chest_xray_report.json")
    parser.add_argument("--output-dir", type=str, default="saved_models/chest_xray/reports", help="Output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        print(json.dumps({"status": "error", "error_message": f"File not found: {image_path}"}, indent=2))
        return 1

    config = ChestXrayConfig.load(args.config)
    ckpt_mgr = CheckpointManager(saved_model_dir=config.model.saved_model_dir)

    label_map_path = Path(config.data.label_map_output_path)
    if not label_map_path.is_file():
        print(json.dumps({"status": "error", "error_message": f"Label map missing at {label_map_path}. Run training first."}, indent=2))
        return 1

    with open(label_map_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
        label_map = meta.get("class_to_label", {})

    num_classes = len(label_map)
    model = build_chest_xray_model(config.model, num_classes=num_classes)

    ckpt_mgr.load_checkpoint(model=model, load_best=True)

    predictor = ChestXrayPredictor(model=model, label_map=label_map)
    result = predictor.predict_image(image_path)

    standard_json = result.to_standard_dict()
    print(json.dumps(standard_json, indent=2))

    if args.report:
        ClinicalReportGenerator.generate_report(result, output_path=Path(args.output_dir) / "chest_xray_report.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
