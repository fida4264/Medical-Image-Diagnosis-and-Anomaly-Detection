"""
Medical AI Engine — Brain MRI Predictor Adapter
Wraps Module 7 (Brain MRI Disease Analysis Framework).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional
from PIL import Image

from brain_mri.inference.predict_brain_mri import BrainMRIPredictor
from medical_ai_engine.adapters.base_predictor_adapter import BasePredictorAdapter


class BrainMRIPredictorAdapter(BasePredictorAdapter):
    """
    Adapter wrapping BrainMRIPredictor (Module 7).
    """

    @property
    def modality(self) -> str:
        return "brain_mri"

    def load_model(self, checkpoint_path: Optional[str] = None, label_map_path: Optional[str] = None) -> None:
        ckpt = checkpoint_path or self._checkpoint_path or "saved_models/brain_mri/best_model.pt"
        lbl_map = label_map_path or self._label_map_path or "metadata/brain_mri_label_map.json"

        self._predictor = BrainMRIPredictor(
            checkpoint_path=ckpt,
            label_map_path=lbl_map,
        )

        self._checkpoint_path = str(ckpt)
        self._label_map_path = str(lbl_map)
        self._is_loaded = True

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        if not self.is_loaded:
            self.load_model()
        assert self._predictor is not None

        res = self._predictor.predict(image)
        res_dict = res.to_dict() if hasattr(res, "to_dict") else res

        predicted_class = res_dict.get("predicted_disease") or res_dict.get("predicted_class", "unknown")
        confidence = float(res_dict.get("confidence", 0.0))

        top_3 = res_dict.get("top_3_predictions", [])
        top_k = []
        for item in top_3:
            if isinstance(item, dict):
                top_k.append({"disease": item.get("disease", "unknown"), "confidence": float(item.get("confidence", 0.0))})
            elif isinstance(item, (list, tuple)):
                top_k.append({"disease": item[0], "confidence": float(item[1])})

        reliability = "high" if confidence >= 0.85 else ("moderate" if confidence >= 0.65 else "low")
        manual_review = confidence < 0.85

        return {
            "module": "brain_mri",
            "modality": "brain_mri",
            "predicted_class": predicted_class,
            "confidence": confidence,
            "top_k_predictions": top_k,
            "top3_predictions": top_k,
            "inference_time_ms": float(res_dict.get("inference_time_ms", 0.0)),
            "clinical_report": res_dict.get("clinical_report", {}),
            "prediction_reliability": reliability,
            "requires_manual_review": manual_review,
        }

    def get_metadata(self) -> Dict[str, Any]:
        meta = super().get_metadata()
        if self._predictor and hasattr(self._predictor, "label_map"):
            meta["supported_classes"] = list(self._predictor.label_map.keys())
        else:
            meta["supported_classes"] = ["glioma", "meningioma", "notumor", "pituitary"]
        return meta
