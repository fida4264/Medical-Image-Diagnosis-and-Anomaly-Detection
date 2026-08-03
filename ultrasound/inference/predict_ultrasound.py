"""
ultrasound/inference/predict_ultrasound.py
============================================
Single-image inference engine for ultrasound disease classification and reliability scoring.
Produces standardized prediction JSON schema compatible with Module 14 Unified Prediction Engine.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from modality_classifier.datasets.data_loader import IMAGENET_MEAN, IMAGENET_STD  # Module 6 reuse

logger = logging.getLogger(__name__)

MODULE_VERSION = "0.13.0"


class PredictionReliabilityScorer:
    """
    Evaluates clinical prediction reliability and determines if manual review is required.
    """

    @staticmethod
    def score(confidence: float, high_thresh: float = 0.85, mod_thresh: float = 0.65) -> Tuple[str, bool]:
        """
        Returns (reliability_str, requires_manual_review_bool).
        """
        if confidence >= high_thresh:
            return "high", False
        elif confidence >= mod_thresh:
            return "moderate", False
        else:
            return "low", True


@dataclass
class UltrasoundPredictionResult:
    """Standardized prediction output for ultrasound disease classification."""

    predicted_class: str
    confidence: float
    probability_distribution: Dict[str, float]
    top_k_predictions: List[Tuple[str, float]]
    prediction_reliability: str
    requires_manual_review: bool
    inference_time_ms: float
    prediction_timestamp: str
    status: str = "success"
    model_version: str = MODULE_VERSION

    def to_standard_dict(self) -> dict:
        """Serialize to Module 14 Unified Prediction Engine JSON schema."""
        return {
            "module": "ultrasound",
            "modality": "ultrasound",
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "prediction_reliability": self.prediction_reliability,
            "requires_manual_review": self.requires_manual_review,
            "top3_predictions": [
                [cls, round(prob, 6)] for cls, prob in self.top_k_predictions[:3]
            ],
            "probability_distribution": {
                k: round(v, 6) for k, v in self.probability_distribution.items()
            },
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "timestamp": self.prediction_timestamp,
            "status": self.status,
        }


class UltrasoundPredictor:
    """Single-image ultrasound disease predictor and reliability scorer."""

    def __init__(
        self,
        model: nn.Module,
        label_map: Dict[str, int],
        target_size: Tuple[int, int] = (224, 224),
        top_k: int = 3,
        high_threshold: float = 0.85,
        moderate_threshold: float = 0.65,
    ):
        self.model = model
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.top_k = top_k
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    @torch.no_grad()
    def predict_image(self, image: Image.Image) -> UltrasoundPredictionResult:
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

        reliability, manual_review = PredictionReliabilityScorer.score(
            confidence, self.high_threshold, self.moderate_threshold
        )

        return UltrasoundPredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probability_distribution=prob_dist,
            top_k_predictions=top_k,
            prediction_reliability=reliability,
            requires_manual_review=manual_review,
            inference_time_ms=inference_time_ms,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            status="success",
            model_version=MODULE_VERSION,
        )

    @torch.no_grad()
    def predict_path(self, image_path: str | Path) -> UltrasoundPredictionResult:
        """Run inference on an image file path."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)
