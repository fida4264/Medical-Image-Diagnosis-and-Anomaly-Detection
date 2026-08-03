"""
predict_ultrasound.py
======================
CLI tool for Module 13: Ultrasound Disease Analysis Framework.
Runs single-image ultrasound disease diagnosis, prediction reliability scoring, and report generation from a trained checkpoint.

Usage:
    python predict_ultrasound.py --image <path_to_image>
    python predict_ultrasound.py --image <path_to_image> --output outputs/us_pred.json
    python predict_ultrasound.py --image <path_to_image> --top-k 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from ultrasound.inference.predict_ultrasound import UltrasoundPredictor
from ultrasound.models.ultrasound_model import build_ultrasound_model
from ultrasound.utils.config import UltrasoundConfig
from ultrasound.utils.logger import get_ultrasound_logger

logger = get_ultrasound_logger("predict_ultrasound")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Module 13: Ultrasound Disease Diagnosis & Prediction Reliability Scoring"
    )
    parser.add_argument("--image", type=str, required=True, help="Path to ultrasound image")
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to trained model checkpoint (.pt). Defaults to saved_models/ultrasound/best_model.pt"
    )
    parser.add_argument(
        "--label-map", type=str, default=None,
        help="Path to label map JSON. Defaults to metadata/ultrasound_label_map.json"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to ultrasound.yaml")
    parser.add_argument("--output", type=str, default=None, help="Path to save prediction JSON")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = UltrasoundConfig.load(args.config)

    # Resolve checkpoint
    checkpoint_path = (
        Path(args.checkpoint) if args.checkpoint
        else Path(config.model.saved_model_dir) / "best_model.pt"
    )
    if not checkpoint_path.is_file():
        logger.error(f"Checkpoint not found: '{checkpoint_path}'. Run run_ultrasound.py first.")
        return 1

    # Resolve label map
    label_map_path = (
        Path(args.label_map) if args.label_map
        else Path(config.data.label_map_output_path)
    )
    if not label_map_path.is_file():
        logger.error(f"Label map not found: '{label_map_path}'. Run run_ultrasound.py first.")
        return 1

    with open(label_map_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    label_map: dict = meta["class_to_label"]
    num_classes = meta["num_classes"]

    # Load model
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_ultrasound_model(config.model, num_classes=num_classes)
    model.load_state_dict(state["model_state_dict"])
    logger.info(f"Model loaded from '{checkpoint_path}'.")

    # Run prediction
    predictor = UltrasoundPredictor(
        model=model,
        label_map=label_map,
        top_k=args.top_k,
        high_threshold=config.reliability.high_threshold,
        moderate_threshold=config.reliability.moderate_threshold,
    )
    result = predictor.predict_path(args.image)

    # Display result
    print("\n" + "=" * 66)
    print("   Module 13 — Ultrasound Disease Prediction & Reliability   ")
    print("=" * 66)
    print(f"  Image                  : {args.image}")
    print(f"  Predicted Class        : {result.predicted_class}")
    print(f"  Prediction Reliability : {result.prediction_reliability}")
    print(f"  Requires Manual Review : {result.requires_manual_review}")
    print(f"  Confidence             : {result.confidence*100:.2f}%")
    print(f"  Inference Time         : {result.inference_time_ms:.2f} ms")
    print(f"\n  Top-{args.top_k} Predictions:")
    for i, (cls, prob) in enumerate(result.top_k_predictions, 1):
        print(f"    {i}. {cls:<35} {prob*100:.2f}%")
    print("=" * 66 + "\n")

    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result.to_standard_dict(), f, indent=2)
        logger.info(f"Prediction saved to '{output_path}'.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
