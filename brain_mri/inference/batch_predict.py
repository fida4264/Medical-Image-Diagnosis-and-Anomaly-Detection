"""
brain_mri/inference/batch_predict.py
====================================
Batch prediction processor for directory of Brain MRI images.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from brain_mri.inference.predict_brain_mri import BrainMRIPredictor

logger = logging.getLogger(__name__)


def run_batch_prediction(
    directory: str | Path,
    predictor: Optional[BrainMRIPredictor] = None,
    output_json: str | Path = "saved_models/brain_mri/reports/batch_predictions.json",
    max_images: int = 100,
) -> Path:
    dir_path = Path(directory)
    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if predictor is None:
        predictor = BrainMRIPredictor()

    valid_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    image_files = [p for p in dir_path.rglob("*") if p.suffix.lower() in valid_exts][:max_images]

    logger.info(f"Running batch prediction on {len(image_files)} Brain MRI images...")
    results = []
    for img_p in image_files:
        res = predictor.predict(img_p)
        results.append(res.to_dict())

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Saved batch prediction results to {out_path}")
    return out_path
