"""
brain_mri/inference/predict_brain_mri.py
=========================================
Brain MRI Disease Inference Predictor & Clinical Report Engine.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from brain_mri.models.brain_mri_model import BrainMRIClassifier
from brain_mri.reports.clinical_report import ClinicalReportData, ClinicalReportGenerator
from modality_classifier.datasets.data_loader import IMAGENET_MEAN, IMAGENET_STD  # REUSE Module 6

logger = logging.getLogger(__name__)


@dataclass
class BrainMRIPredictionResult:
    predicted_disease: str
    confidence: float
    top_3_predictions: List[Dict[str, Any]]
    inference_time_ms: float
    clinical_report: Dict[str, Any]
    image_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BrainMRIPredictor:
    """
    Inference predictor for Brain MRI Disease Classification.
    """

    def __init__(
        self,
        checkpoint_path: str | Path = "saved_models/brain_mri/best_model.pt",
        label_map_path: str | Path = "metadata/brain_mri_label_map.json",
        device: Optional[str] = None,
        target_size: Tuple[int, int] = (224, 224),
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.label_map_path = Path(label_map_path)
        self.target_size = target_size

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.label_map, self.inv_label_map = self._load_label_map()
        self.model = self._load_model()
        self.report_generator = ClinicalReportGenerator()

        self.transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _load_label_map(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        if not self.label_map_path.is_file():
            default_classes = ["glioma", "meningioma", "notumor", "pituitary"]
            l_map = {c: i for i, c in enumerate(default_classes)}
            inv_map = {i: c for c, i in l_map.items()}
            return l_map, inv_map

        with open(self.label_map_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        l_map = data.get("class_to_label", {})
        inv_map = {int(k): v for k, v in data.get("label_to_class", {}).items()}
        return l_map, inv_map

    def _load_model(self) -> BrainMRIClassifier:
        num_classes = len(self.inv_label_map)
        if not self.checkpoint_path.is_file():
            logger.warning(f"Checkpoint {self.checkpoint_path} not found. Initializing base model.")
            model = BrainMRIClassifier(num_classes=num_classes)
            model.to(self.device)
            model.eval()
            return model

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        backbone = checkpoint.get("backbone", "efficientnet_b0")

        model = BrainMRIClassifier(num_classes=num_classes, backbone_name=backbone)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        logger.info(f"Loaded BrainMRIClassifier checkpoint from {self.checkpoint_path}")
        return model

    def _preprocess_input(self, image_input: Union[str, Path, Image.Image, np.ndarray]) -> torch.Tensor:
        if isinstance(image_input, (str, Path)):
            path = Path(image_input)
            if not path.is_file():
                raise FileNotFoundError(f"Image file not found: {path}")
            img = Image.open(path).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        elif isinstance(image_input, np.ndarray):
            img = Image.fromarray(image_input.astype(np.uint8)).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        return self.transform(img).unsqueeze(0)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        top_k: int = 3,
    ) -> BrainMRIPredictionResult:
        """
        Runs disease prediction for a single Brain MRI image and generates clinical report.
        """
        t0 = time.perf_counter()

        tensor_img = self._preprocess_input(image_input).to(self.device)
        logits = self.model(tensor_img)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top_indices = np.argsort(probs)[::-1][:top_k]
        top_3_preds = [
            {
                "disease": self.inv_label_map.get(int(idx), f"Class_{idx}"),
                "confidence": float(probs[idx]),
            }
            for idx in top_indices
        ]

        pred_disease = top_3_preds[0]["disease"]
        confidence = top_3_preds[0]["confidence"]

        prob_dist = {
            self.inv_label_map.get(i, f"Class_{i}"): float(probs[i])
            for i in range(len(probs))
        }

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        img_path_str = str(image_input) if isinstance(image_input, (str, Path)) else None

        # Generate Clinical Report
        report = self.report_generator.generate_report(
            predicted_disease=pred_disease,
            confidence=confidence,
            probability_distribution=prob_dist,
            top_3_predictions=top_3_preds,
            inference_time_ms=elapsed_ms,
            image_path=img_path_str,
        )

        return BrainMRIPredictionResult(
            predicted_disease=pred_disease,
            confidence=confidence,
            top_3_predictions=top_3_preds,
            inference_time_ms=round(elapsed_ms, 2),
            clinical_report=report.to_dict(),
            image_path=img_path_str,
        )
