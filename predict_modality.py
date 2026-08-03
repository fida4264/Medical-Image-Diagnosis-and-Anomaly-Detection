"""
predict_modality.py
===================
CLI entrypoint for image modality inference prediction and routing.

Usage
-----
    python predict_modality.py --image processed_images/brain_mri/glioma/MRI_000001.png
    python predict_modality.py --dir datasets/chest_xray/Pneumonia/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from modality_classifier.inference.predictor import ModalityPredictor
from modality_classifier.utils.logging import setup_logger

logger = setup_logger("predict_modality")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Medical Image Modality & Routing")
    parser.add_argument("--image", type=str, default=None, help="Path to single medical image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of image files")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to report")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/modality_classifier_best.pt", help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image and not args.dir:
        logger.error("Must provide either --image <path> or --dir <path>.")
        sys.exit(1)

    predictor = ModalityPredictor(checkpoint_path=args.checkpoint)

    if args.image:
        result = predictor.predict(args.image, top_k=args.top_k)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.dir:
        dir_path = Path(args.dir)
        if not dir_path.is_dir():
            logger.error(f"Directory not found: {dir_path}")
            sys.exit(1)

        valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
        image_files = [p for p in dir_path.rglob("*") if p.suffix.lower() in valid_exts]

        logger.info(f"Processing batch of {len(image_files)} images from {dir_path}...")
        batch_results = []
        for img_p in image_files[:50]:  # limit sample preview to 50
            res = predictor.predict(img_p, top_k=args.top_k)
            batch_results.append(res.to_dict())

        output_path = Path("outputs/batch_modality_predictions.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, indent=2)

        logger.info(f"Batch inference complete! Saved {len(batch_results)} predictions to {output_path}")


if __name__ == "__main__":
    main()
