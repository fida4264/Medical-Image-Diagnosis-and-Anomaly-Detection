"""
tests/test_ultrasound.py
=========================
Comprehensive unit tests for Module 13: Ultrasound Disease Analysis Framework.
Mirrors the exact structure of test_brain_mri.py, test_chest_xray.py, test_histopathology.py, test_retina.py, test_skin.py.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch


# ---------------------------------------------------------------------------
# 1. Config Loading
# ---------------------------------------------------------------------------
class TestUltrasoundConfig:
    def test_default_config_loads(self):
        from ultrasound.utils.config import UltrasoundConfig
        cfg = UltrasoundConfig.load()
        assert cfg.model.backbone == "efficientnet_b0"
        assert cfg.training.epochs > 0
        assert cfg.data.target_size == [224, 224]
        assert cfg.reliability.high_threshold == 0.85

    def test_config_missing_file_uses_defaults(self):
        from ultrasound.utils.config import UltrasoundConfig
        cfg = UltrasoundConfig.load("/nonexistent/path/ultrasound.yaml")
        assert cfg.model.dropout_rate > 0
        assert cfg.training.batch_size > 0

    def test_split_ratios_sum_to_one(self):
        from ultrasound.utils.config import UltrasoundConfig
        cfg = UltrasoundConfig.load()
        ratios = cfg.data.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Dataset Loader — strict modality filter
# ---------------------------------------------------------------------------
class TestUltrasoundDatasetLoader:
    def _make_df(self, num_classes: int = 3, rows_per_class: int = 10) -> pd.DataFrame:
        classes = ["benign", "malignant", "normal"][:num_classes]
        rows = []
        for c in classes:
            for j in range(rows_per_class):
                rows.append({
                    "modality": "ultrasound",
                    "normalized_class": c,
                    "internal_id": f"{c}_{j}",
                    "is_valid": True,
                    "split": "unspecified",
                })
        # non-ultrasound rows — must be filtered out
        rows.append({"modality": "brain_mri", "normalized_class": "glioma",
                     "internal_id": "brain_0", "is_valid": True, "split": "train"})
        rows.append({"modality": "skin", "normalized_class": "melanoma",
                     "internal_id": "skin_0", "is_valid": True, "split": "train"})
        return pd.DataFrame(rows)

    def test_strict_modality_filter(self, tmp_path):
        from ultrasound.datasets.dataset_loader import UltrasoundDatasetLoader
        df = self._make_df()
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = UltrasoundDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        filtered = loader.load_and_filter()
        assert (filtered["modality"] == "ultrasound").all()
        assert len(filtered) == 30  # 3 classes × 10 rows

    def test_dynamic_class_discovery(self, tmp_path):
        from ultrasound.datasets.dataset_loader import UltrasoundDatasetLoader
        df = self._make_df(num_classes=3)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = UltrasoundDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        loader.load_and_filter()
        assert len(loader.label_map) == 3
        assert set(loader.label_map.keys()) == {"benign", "malignant", "normal"}

    def test_prepare_splits_creates_files(self, tmp_path):
        from ultrasound.datasets.dataset_loader import UltrasoundDatasetLoader
        df = self._make_df(num_classes=3, rows_per_class=30)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        splits_path = tmp_path / "splits.csv"
        label_map_path = tmp_path / "label_map.json"

        loader = UltrasoundDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=splits_path,
            label_map_output_path=label_map_path,
        )
        loader.load_and_filter()
        splits_df = loader.prepare_splits()

        assert splits_path.is_file()
        assert label_map_path.is_file()
        assert "ultrasound_split" in splits_df.columns
        assert set(splits_df["ultrasound_split"].unique()) == {"train", "val", "test"}

        with open(label_map_path) as f:
            meta = json.load(f)
        assert meta["modality"] == "ultrasound"
        assert meta["num_classes"] == 3


# ---------------------------------------------------------------------------
# 3. Model — build and forward pass
# ---------------------------------------------------------------------------
class TestUltrasoundModel:
    def test_model_builds_and_forward_pass(self):
        from ultrasound.models.ultrasound_model import build_ultrasound_model
        from ultrasound.utils.config import ModelConfig

        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False, dropout_rate=0.3)
        model = build_ultrasound_model(cfg, num_classes=3)
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 3)

    def test_num_classes_matches_output(self):
        from ultrasound.models.ultrasound_model import build_ultrasound_model
        from ultrasound.utils.config import ModelConfig

        for n in [2, 3, 5]:
            cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
            model = build_ultrasound_model(cfg, num_classes=n)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            assert out.shape[1] == n


# ---------------------------------------------------------------------------
# 4. PredictionReliabilityScorer
# ---------------------------------------------------------------------------
class TestPredictionReliabilityScorer:
    def test_high_reliability(self):
        from ultrasound.inference.predict_ultrasound import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.92)
        assert reliability == "high"
        assert review is False

    def test_moderate_reliability(self):
        from ultrasound.inference.predict_ultrasound import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.75)
        assert reliability == "moderate"
        assert review is False

    def test_low_reliability_triggers_manual_review(self):
        from ultrasound.inference.predict_ultrasound import PredictionReliabilityScorer
        reliability, review = PredictionReliabilityScorer.score(0.50)
        assert reliability == "low"
        assert review is True


# ---------------------------------------------------------------------------
# 5. Predictor — output schema and reliability integration
# ---------------------------------------------------------------------------
class TestUltrasoundPredictor:
    def test_predictor_output_schema(self):
        from PIL import Image
        from ultrasound.inference.predict_ultrasound import UltrasoundPredictor
        from ultrasound.models.ultrasound_model import build_ultrasound_model
        from ultrasound.utils.config import ModelConfig

        label_map = {"benign": 0, "malignant": 1, "normal": 2}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_ultrasound_model(cfg, num_classes=3)

        predictor = UltrasoundPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224), color=(120, 120, 120))
        result = predictor.predict_image(dummy_img)

        assert result.predicted_class in label_map
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top_k_predictions) <= 3
        assert abs(sum(result.probability_distribution.values()) - 1.0) < 1e-4
        assert result.inference_time_ms > 0
        assert result.prediction_reliability in ["high", "moderate", "low"]
        assert isinstance(result.requires_manual_review, bool)

    def test_to_standard_dict_schema(self):
        from PIL import Image
        from ultrasound.inference.predict_ultrasound import UltrasoundPredictor
        from ultrasound.models.ultrasound_model import build_ultrasound_model
        from ultrasound.utils.config import ModelConfig

        label_map = {"benign": 0, "malignant": 1, "normal": 2}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_ultrasound_model(cfg, num_classes=3)

        predictor = UltrasoundPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224))
        result = predictor.predict_image(dummy_img)
        d = result.to_standard_dict()

        required_keys = {
            "module", "modality", "predicted_class", "confidence",
            "probability_distribution", "top3_predictions",
            "prediction_reliability", "requires_manual_review",
            "model_version", "inference_time_ms", "timestamp", "status",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["module"] == "ultrasound"
        assert d["modality"] == "ultrasound"
        assert d["status"] == "success"
        assert d["prediction_reliability"] in ["high", "moderate", "low"]
        assert isinstance(d["requires_manual_review"], bool)


# ---------------------------------------------------------------------------
# 6. CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManager:
    def test_save_and_load_best(self, tmp_path):
        from ultrasound.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}, "val_accuracy": 0.92}
        manager.save_best(state)
        assert manager.has_best()
        loaded = manager.load_best()
        assert loaded is not None
        assert loaded["val_accuracy"] == pytest.approx(0.92)

    def test_save_and_load_last(self, tmp_path):
        from ultrasound.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"epoch": 7, "model_state_dict": {}}
        manager.save_last(state)
        assert manager.has_last()
        loaded = manager.load_last()
        assert loaded["epoch"] == 7


# ---------------------------------------------------------------------------
# 7. Package __init__ public API
# ---------------------------------------------------------------------------
class TestUltrasoundPackageInit:
    def test_public_api_importable(self):
        import ultrasound
        expected = [
            "UltrasoundDatasetLoader",
            "UltrasoundDataset",
            "create_ultrasound_data_loaders",
            "UltrasoundClassifier",
            "build_ultrasound_model",
            "UltrasoundTrainer",
            "UltrasoundEvaluator",
            "UltrasoundPredictor",
            "UltrasoundPredictionResult",
            "PredictionReliabilityScorer",
            "ClinicalReportGenerator",
            "UltrasoundConfig",
        ]
        for name in expected:
            assert hasattr(ultrasound, name), f"Missing from ultrasound.__init__: {name}"

    def test_version_string(self):
        import ultrasound
        assert ultrasound.__version__ == "0.13.0"
