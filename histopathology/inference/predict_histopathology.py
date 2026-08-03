"""
histopathology/inference/predict_histopathology.py
====================================================
Single-image inference engine for histopathology cancer subtyping.
Produces a standardized prediction JSON schema compatible with Module 14
Unified Prediction Engine. Mirrors Module 7/8/9 predictor pattern.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from modality_classifier.datasets.data_loader import IMAGENET_MEAN, IMAGENET_STD  # Module 6 reuse

logger = logging.getLogger(__name__)

MODULE_VERSION = "0.10.0"


@dataclass
class HistopathologyPredictionResult:
    """Standardized prediction output for histopathology cancer subtyping."""

    predicted_class: str
    confidence: float
    probability_distribution: Dict[str, float]
    top_k_predictions: List[Tuple[str, float]]
    inference_time_ms: float
    prediction_timestamp: str
    model_version: str = MODULE_VERSION

    def to_standard_dict(self) -> dict:
        """Serialize to the Module 14 Unified Prediction Engine JSON schema."""
        return {
            "module": "histopathology",
            "modality": "histopathology",
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "probability_distribution": {
                k: round(v, 6) for k, v in self.probability_distribution.items()
            },
            "top_3_predictions": [
                [cls, round(prob, 6)] for cls, prob in self.top_k_predictions[:3]
            ],
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "prediction_timestamp": self.prediction_timestamp,
        }


class HistopathologyPredictor:
    """Single-image histopathology cancer subtype predictor."""

    def __init__(
        self,
        model: nn.Module,
        label_map: Dict[str, int],
        target_size: Tuple[int, int] = (224, 224),
        top_k: int = 3,
    ):
        self.model = model
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.top_k = top_k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> HistopathologyPredictionResult:
        """Run inference on a single PIL image and return a prediction result."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        t0 = time.perf_counter()
        logits = self.model(tensor)
        inference_time_ms = (time.perf_counter() - t0) * 1000

        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
        pred_idx = int(torch.argmax(torch.tensor(probs)).item())
        pred_class = self.inv_label_map.get(pred_idx, "unknown")
        confidence = probs[pred_idx]

        prob_dist = {
            self.inv_label_map.get(i, str(i)): float(p)
            for i, p in enumerate(probs)
        }

        sorted_probs = sorted(prob_dist.items(), key=lambda x: x[1], reverse=True)
        top_k = sorted_probs[:self.top_k]

        return HistopathologyPredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probability_distribution=prob_dist,
            top_k_predictions=top_k,
            inference_time_ms=inference_time_ms,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=MODULE_VERSION,
        )

    @torch.no_grad()
    def predict_path(self, image_path: str | Path) -> HistopathologyPredictionResult:
        """Run inference on an image file path."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)
