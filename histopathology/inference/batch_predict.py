"""
histopathology/inference/batch_predict.py
==========================================
Batch CSV inference engine for histopathology cancer subtyping.
Mirrors Module 7/8/9 batch_predict.py pattern.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch.nn as nn
from PIL import Image

from histopathology.inference.predict_histopathology import HistopathologyPredictor

logger = logging.getLogger(__name__)


class BatchHistopathologyPredictor:
    """
    Runs histopathology predictions on a list of image paths and saves
    results to a CSV file.
    """

    def __init__(
        self,
        model: nn.Module,
        label_map: Dict[str, int],
        top_k: int = 3,
    ):
        self.predictor = HistopathologyPredictor(model=model, label_map=label_map, top_k=top_k)

    def predict_batch(
        self,
        image_paths: List[str | Path],
        output_csv: Optional[str | Path] = None,
    ) -> List[dict]:
        """
        Run predictions on a list of image paths.
        Optionally save results to a CSV file.
        """
        results = []
        for img_path in image_paths:
            try:
                result = self.predictor.predict_path(img_path)
                row = result.to_standard_dict()
                row["image_path"] = str(img_path)
                results.append(row)
                logger.info(
                    f"[{img_path}] → {result.predicted_class} "
                    f"({result.confidence*100:.1f}%)"
                )
            except Exception as e:
                logger.error(f"Prediction failed for '{img_path}': {e}")
                results.append({"image_path": str(img_path), "error": str(e)})

        if output_csv:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            if results:
                fieldnames = list(results[0].keys())
                with open(output_csv, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(results)
            logger.info(f"Batch predictions saved to '{output_csv}'.")

        return results
