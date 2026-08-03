"""
chest_xray/inference/batch_predict.py
======================================
Batch inference engine for processing multiple Chest X-ray images simultaneously.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image

from chest_xray.inference.predict_chest_xray import ChestXrayPredictionResult, ChestXrayPredictor

logger = logging.getLogger(__name__)


def run_batch_inference(
    predictor: ChestXrayPredictor,
    image_paths: List[str | Path],
    output_json_path: Optional[str | Path] = None,
) -> List[ChestXrayPredictionResult]:
    """Runs inference across a list of image file paths."""
    results = []
    logger.info(f"Running batch inference on {len(image_paths)} images...")

    for path in image_paths:
        p = Path(path)
        if not p.is_file():
            logger.warning(f"File not found, skipping: {p}")
            continue
        try:
            res = predictor.predict_image(p)
            results.append(res)
        except Exception as e:
            logger.error(f"Error predicting {p}: {e}")
            results.append(
                ChestXrayPredictionResult(
                    predicted_class="unknown",
                    confidence=0.0,
                    top3_predictions=[],
                    probabilities={},
                    image_path=str(p),
                    status="failed",
                    error_message=str(e),
                )
            )

    if output_json_path:
        out_p = Path(output_json_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        serialized = [r.to_standard_dict() for r in results]
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(serialized, f, indent=2)
        logger.info(f"Saved batch prediction results to {out_p}")

    return results
