"""
skin/inference/batch_predict.py
================================
Batch inference runner for Module 12: Skin Lesion Cancer Classification Framework.
Processes a directory of images and exports batch predictions to CSV/JSON.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from PIL import Image

from skin.inference.predict_skin import SkinLesionPredictor

logger = logging.getLogger(__name__)


class BatchSkinPredictor:
    """Runs batch inference over a folder of skin lesion images."""

    def __init__(self, predictor: SkinLesionPredictor):
        self.predictor = predictor

    def predict_directory(
        self,
        image_dir: str | Path,
        output_csv_path: Optional[str | Path] = None,
        extensions: tuple = (".jpg", ".jpeg", ".png", ".tif", ".bmp"),
    ) -> pd.DataFrame:
        """Run inference on all image files in image_dir."""
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            raise NotADirectoryError(f"Directory not found: {image_dir}")

        image_paths = [p for p in image_dir.rglob("*") if p.suffix.lower() in extensions]
        logger.info(f"Found {len(image_paths)} image files in '{image_dir}'. Running batch inference...")

        results = []
        for path in image_paths:
            try:
                res = self.predictor.predict_path(path)
                results.append({
                    "image_path": str(path),
                    "filename": path.name,
                    "predicted_class": res.predicted_class,
                    "confidence": res.confidence,
                    "prediction_reliability": res.prediction_reliability,
                    "requires_manual_review": res.requires_manual_review,
                    "inference_time_ms": res.inference_time_ms,
                })
            except Exception as e:
                logger.error(f"Failed to process image '{path}': {e}")

        df = pd.DataFrame(results)

        if output_csv_path:
            output_csv_path = Path(output_csv_path)
            output_csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Batch prediction results written to '{output_csv_path}'.")

        return df
