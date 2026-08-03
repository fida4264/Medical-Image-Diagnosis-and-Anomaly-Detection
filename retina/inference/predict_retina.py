"""
retina/inference/predict_retina.py
===================================
Single-image inference engine for retina disease diagnosis and severity grading.
Produces standardized prediction JSON schema compatible with Module 14 Unified Prediction Engine.
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

MODULE_VERSION = "0.11.0"


class SeverityGrader:
    """
    Evaluates and grades disease severity for retinal findings.
    Supports grading based on discovered class names, stage keywords, and prediction confidence.
    """

    @staticmethod
    def grade(predicted_class: str, confidence: float) -> str:
        cls_lower = predicted_class.lower()

        # Direct stage keywords in class names
        if "normal" in cls_lower or "no_dr" in cls_lower:
            return "Grade 0: Normal / No Apparent Retinopathy"
        elif "mild" in cls_lower or "stage1" in cls_lower or "stage_1" in cls_lower:
            return "Grade 1: Mild Retinopathy"
        elif "moderate" in cls_lower or "stage2" in cls_lower or "stage_2" in cls_lower:
            return "Grade 2: Moderate Retinopathy"
        elif "severe" in cls_lower or "stage3" in cls_lower or "stage_3" in cls_lower:
            return "Grade 3: Severe Retinopathy"
        elif "proliferative" in cls_lower or "stage4" in cls_lower or "stage_4" in cls_lower or "advanced" in cls_lower:
            return "Grade 4: Proliferative / High-Risk Retinopathy"

        # Confidence-based severity grading for un-staged retinal findings
        if confidence >= 0.90:
            return "High Confidence Finding (Stage 2/3 - Clinical Correlation Required)"
        elif confidence >= 0.70:
            return "Moderate Confidence Finding (Stage 1/2 - Follow-up Recommended)"
        else:
            return "Ungraded / Early Finding (Low Confidence - Screening Recommended)"


@dataclass
class RetinaPredictionResult:
    """Standardized prediction output for retina disease diagnosis and severity grading."""

    predicted_class: str
    confidence: float
    probability_distribution: Dict[str, float]
    top_k_predictions: List[Tuple[str, float]]
    severity: str
    inference_time_ms: float
    prediction_timestamp: str
    status: str = "success"
    model_version: str = MODULE_VERSION

    def to_standard_dict(self) -> dict:
        """Serialize to Module 14 Unified Prediction Engine JSON schema."""
        return {
            "module": "retina",
            "modality": "retina",
            "predicted_class": self.predicted_class,
            "confidence": round(self.confidence, 6),
            "top3_predictions": [
                [cls, round(prob, 6)] for cls, prob in self.top_k_predictions[:3]
            ],
            "severity": self.severity,
            "probability_distribution": {
                k: round(v, 6) for k, v in self.probability_distribution.items()
            },
            "model_version": self.model_version,
            "inference_time_ms": round(self.inference_time_ms, 3),
            "timestamp": self.prediction_timestamp,
            "status": self.status,
        }


class RetinaPredictor:
    """Single-image retina disease predictor and severity grader."""

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
    def predict_image(self, image: Image.Image) -> RetinaPredictionResult:
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

        severity = SeverityGrader.grade(pred_class, confidence)

        return RetinaPredictionResult(
            predicted_class=pred_class,
            confidence=confidence,
            probability_distribution=prob_dist,
            top_k_predictions=top_k,
            severity=severity,
            inference_time_ms=inference_time_ms,
            prediction_timestamp=datetime.now(timezone.utc).isoformat(),
            status="success",
            model_version=MODULE_VERSION,
        )

    @torch.no_grad()
    def predict_path(self, image_path: str | Path) -> RetinaPredictionResult:
        """Run inference on an image file path."""
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        return self.predict_image(image)
