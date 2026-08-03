"""
predict_brain_mri.py
====================
CLI entrypoint for running single-image or batch Brain MRI disease prediction.

Usage
-----
    python predict_brain_mri.py --image processed_images/brain_mri/glioma/MRI_000001.png
    python predict_brain_mri.py --dir datasets/brain_mri/Glioma/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from brain_mri.inference.batch_predict import run_batch_prediction
from brain_mri.inference.predict_brain_mri import BrainMRIPredictor
from brain_mri.utils.logger import get_brain_mri_logger

logger = get_brain_mri_logger("predict_brain_mri")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Brain MRI Disease & Generate Clinical Report")
    parser.add_argument("--image", type=str, default=None, help="Path to single Brain MRI image file")
    parser.add_argument("--dir", type=str, default=None, help="Path to directory of Brain MRI image files")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to include")
    parser.add_argument("--checkpoint", type=str, default="saved_models/brain_mri/best_model.pt", help="Checkpoint path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image and not args.dir:
        logger.error("Must provide either --image <path> or --dir <path>.")
        sys.exit(1)

    predictor = BrainMRIPredictor(checkpoint_path=args.checkpoint)

    if args.image:
        result = predictor.predict(args.image, top_k=args.top_k)
        print(json.dumps(result.to_dict(), indent=2))

    elif args.dir:
        out_path = run_batch_prediction(directory=args.dir, predictor=predictor)
        logger.info(f"Batch prediction finished. Saved results to {out_path}")


if __name__ == "__main__":
    main()
