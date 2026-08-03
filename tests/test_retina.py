"""
tests/test_retina.py
======================
Comprehensive unit tests for Module 11: Retina Disease Analysis Framework.
Mirrors the exact structure of test_brain_mri.py, test_chest_xray.py, test_histopathology.py.
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
class TestRetinaConfig:
    def test_default_config_loads(self):
        from retina.utils.config import RetinaConfig
        cfg = RetinaConfig.load()
        assert cfg.model.backbone == "efficientnet_b0"
        assert cfg.training.epochs > 0
        assert cfg.data.target_size == [224, 224]

    def test_config_missing_file_uses_defaults(self):
        from retina.utils.config import RetinaConfig
        cfg = RetinaConfig.load("/nonexistent/path/retina.yaml")
        assert cfg.model.dropout_rate > 0
        assert cfg.training.batch_size > 0

    def test_split_ratios_sum_to_one(self):
        from retina.utils.config import RetinaConfig
        cfg = RetinaConfig.load()
        ratios = cfg.data.split_ratios
        assert abs(sum(ratios.values()) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# 2. Dataset Loader — strict modality filter
# ---------------------------------------------------------------------------
class TestRetinaDatasetLoader:
    def _make_df(self, num_classes: int = 3, rows_per_class: int = 10) -> pd.DataFrame:
        classes = [f"retina_class_{i}" for i in range(num_classes)]
        rows = []
        for c in classes:
            for j in range(rows_per_class):
                rows.append({
                    "modality": "retina",
                    "normalized_class": c,
                    "internal_id": f"{c}_{j}",
                    "is_valid": True,
                    "split": "unspecified",
                })
        # non-retina rows — must be filtered out
        rows.append({"modality": "brain_mri", "normalized_class": "glioma",
                     "internal_id": "brain_0", "is_valid": True, "split": "train"})
        rows.append({"modality": "histopathology", "normalized_class": "colon_normal",
                     "internal_id": "histo_0", "is_valid": True, "split": "train"})
        return pd.DataFrame(rows)

    def test_strict_modality_filter(self, tmp_path):
        from retina.datasets.dataset_loader import RetinaDatasetLoader
        df = self._make_df()
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = RetinaDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        filtered = loader.load_and_filter()
        assert (filtered["modality"] == "retina").all()
        assert len(filtered) == 30  # 3 classes × 10 rows

    def test_dynamic_class_discovery(self, tmp_path):
        from retina.datasets.dataset_loader import RetinaDatasetLoader
        df = self._make_df(num_classes=5)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        loader = RetinaDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=tmp_path / "splits.csv",
            label_map_output_path=tmp_path / "label_map.json",
        )
        loader.load_and_filter()
        assert len(loader.label_map) == 5
        assert all(isinstance(v, int) for v in loader.label_map.values())

    def test_prepare_splits_creates_files(self, tmp_path):
        from retina.datasets.dataset_loader import RetinaDatasetLoader
        df = self._make_df(num_classes=3, rows_per_class=30)
        csv_path = tmp_path / "meta.csv"
        df.to_csv(csv_path, index=False)

        splits_path = tmp_path / "splits.csv"
        label_map_path = tmp_path / "label_map.json"

        loader = RetinaDatasetLoader(
            metadata_path=csv_path,
            processed_dir=tmp_path,
            splits_output_path=splits_path,
            label_map_output_path=label_map_path,
        )
        loader.load_and_filter()
        splits_df = loader.prepare_splits()

        assert splits_path.is_file()
        assert label_map_path.is_file()
        assert "retina_split" in splits_df.columns
        assert set(splits_df["retina_split"].unique()) == {"train", "val", "test"}

        with open(label_map_path) as f:
            meta = json.load(f)
        assert meta["modality"] == "retina"
        assert meta["num_classes"] == 3


# ---------------------------------------------------------------------------
# 3. Model — build and forward pass
# ---------------------------------------------------------------------------
class TestRetinaModel:
    def test_model_builds_and_forward_pass(self):
        from retina.models.retina_model import build_retina_model
        from retina.utils.config import ModelConfig

        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False, dropout_rate=0.3)
        model = build_retina_model(cfg, num_classes=4)
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 4)

    def test_num_classes_matches_output(self):
        from retina.models.retina_model import build_retina_model
        from retina.utils.config import ModelConfig

        for n in [2, 4, 6]:
            cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
            model = build_retina_model(cfg, num_classes=n)
            model.eval()
            with torch.no_grad():
                out = model(torch.randn(1, 3, 224, 224))
            assert out.shape[1] == n


# ---------------------------------------------------------------------------
# 4. SeverityGrader
# ---------------------------------------------------------------------------
class TestSeverityGrader:
    def test_normal_grade_zero(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("normal", 0.99)
        assert "Grade 0" in grade or "Normal" in grade

    def test_mild_grade_one(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("mild_dr", 0.85)
        assert "Grade 1" in grade or "Mild" in grade

    def test_moderate_grade_two(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("moderate_dr", 0.75)
        assert "Grade 2" in grade or "Moderate" in grade

    def test_severe_grade_three(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("severe_dr", 0.90)
        assert "Grade 3" in grade or "Severe" in grade

    def test_proliferative_grade_four(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("proliferative_dr", 0.95)
        assert "Grade 4" in grade or "Proliferative" in grade

    def test_unknown_class_high_confidence(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("diabetic_retinopathy", 0.95)
        assert grade  # should return a non-empty string

    def test_unknown_class_low_confidence(self):
        from retina.inference.predict_retina import SeverityGrader
        grade = SeverityGrader.grade("diabetic_retinopathy", 0.50)
        assert grade  # should return a non-empty string


# ---------------------------------------------------------------------------
# 5. Predictor — output schema and severity integration
# ---------------------------------------------------------------------------
class TestRetinaPredictor:
    def test_predictor_output_schema(self):
        from PIL import Image
        from retina.inference.predict_retina import RetinaPredictor
        from retina.models.retina_model import build_retina_model
        from retina.utils.config import ModelConfig

        label_map = {"normal": 0, "diabetic_retinopathy": 1, "glaucoma": 2, "amd": 3}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_retina_model(cfg, num_classes=4)

        predictor = RetinaPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224), color=(180, 120, 80))
        result = predictor.predict_image(dummy_img)

        assert result.predicted_class in label_map
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.top_k_predictions) <= 3
        assert abs(sum(result.probability_distribution.values()) - 1.0) < 1e-4
        assert result.inference_time_ms > 0
        assert result.severity  # non-empty severity string

    def test_to_standard_dict_schema(self):
        from PIL import Image
        from retina.inference.predict_retina import RetinaPredictor
        from retina.models.retina_model import build_retina_model
        from retina.utils.config import ModelConfig

        label_map = {"normal": 0, "diabetic_retinopathy": 1, "glaucoma": 2}
        cfg = ModelConfig(backbone="efficientnet_b0", pretrained=False)
        model = build_retina_model(cfg, num_classes=3)

        predictor = RetinaPredictor(model=model, label_map=label_map, top_k=3)
        dummy_img = Image.new("RGB", (224, 224))
        result = predictor.predict_image(dummy_img)
        d = result.to_standard_dict()

        required_keys = {
            "module", "modality", "predicted_class", "confidence",
            "probability_distribution", "top3_predictions",
            "severity", "model_version", "inference_time_ms", "timestamp", "status",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["module"] == "retina"
        assert d["modality"] == "retina"
        assert d["status"] == "success"
        assert d["severity"]  # severity field must be non-empty


# ---------------------------------------------------------------------------
# 6. CheckpointManager
# ---------------------------------------------------------------------------
class TestCheckpointManager:
    def test_save_and_load_best(self, tmp_path):
        from retina.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"model_state_dict": {"w": torch.tensor([1.0, 2.0])}, "val_accuracy": 0.92}
        manager.save_best(state)
        assert manager.has_best()
        loaded = manager.load_best()
        assert loaded is not None
        assert loaded["val_accuracy"] == pytest.approx(0.92)

    def test_save_and_load_last(self, tmp_path):
        from retina.utils.checkpoint_manager import CheckpointManager
        manager = CheckpointManager(save_dir=tmp_path)
        state = {"epoch": 7, "model_state_dict": {}}
        manager.save_last(state)
        assert manager.has_last()
        loaded = manager.load_last()
        assert loaded["epoch"] == 7


# ---------------------------------------------------------------------------
# 7. Package __init__ public API
# ---------------------------------------------------------------------------
class TestRetinaPackageInit:
    def test_public_api_importable(self):
        import retina
        expected = [
            "RetinaDatasetLoader",
            "RetinaDataset",
            "create_retina_data_loaders",
            "RetinaClassifier",
            "build_retina_model",
            "RetinaTrainer",
            "RetinaEvaluator",
            "RetinaPredictor",
            "RetinaPredictionResult",
            "SeverityGrader",
            "ClinicalReportGenerator",
            "RetinaConfig",
        ]
        for name in expected:
            assert hasattr(retina, name), f"Missing from retina.__init__: {name}"

    def test_version_string(self):
        import retina
        assert retina.__version__ == "0.11.0"
