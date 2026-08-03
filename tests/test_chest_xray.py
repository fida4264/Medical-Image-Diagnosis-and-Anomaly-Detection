"""
tests/test_chest_xray.py
========================
Comprehensive unit tests for Module 8: Chest X-ray Disease Analysis Framework.
Tests:
1. Dataset Loader & Dynamic Disease Class Discovery
2. Data Builder & PyTorch Transforms
3. ChestXrayClassifier Model Architecture & Feature Extraction
4. Trainer Epoch Execution & Callback Mechanisms
5. Evaluator, ROC, Confusion Matrix, and Benchmarking
6. Predictor, Standardized Prediction JSON Schema, & Clinical Reports
7. ONNX & TorchScript Exporters
8. Configuration Loading
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from chest_xray.datasets.dataset_builder import ChestXrayDataset, create_chest_xray_data_loaders
from chest_xray.datasets.dataset_loader import ChestXrayDatasetLoader
from chest_xray.evaluation.evaluator import ChestXrayEvaluator
from chest_xray.export.onnx_export import ChestXrayONNXExporter
from chest_xray.export.torchscript_export import TorchScriptExporter
from chest_xray.inference.predict_chest_xray import ChestXrayPredictor
from chest_xray.models.chest_xray_model import ChestXrayClassifier, build_chest_xray_model
from chest_xray.reports.clinical_report import ClinicalReportGenerator
from chest_xray.training.trainer import ChestXrayTrainer
from chest_xray.utils.config import ChestXrayConfig


@pytest.fixture
def mock_metadata(tmp_path: Path) -> Path:
    meta_path = tmp_path / "master_metadata.csv"
    data = {
        "internal_id": [f"CX_{i:03d}" for i in range(20)],
        "modality": ["chest_xray"] * 18 + ["brain_mri"] * 2,
        "normalized_class": ["normal", "covid", "pneumonia"] * 6 + ["glioma", "meningioma"],
        "is_valid": [True] * 20,
        "split": ["unspecified"] * 18 + ["train", "train"],
    }
    pd.DataFrame(data).to_csv(meta_path, index=False)
    return meta_path


def test_chest_xray_config_load():
    cfg = ChestXrayConfig.load()
    assert cfg.data.modality_filter == "chest_xray"
    assert cfg.model.backbone == "efficientnet_b0"
    assert cfg.inference.model_version == "0.8.0"


def test_dataset_loader_dynamic_discovery(mock_metadata: Path, tmp_path: Path):
    loader = ChestXrayDatasetLoader(
        metadata_path=mock_metadata,
        processed_dir=tmp_path / "processed",
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    df = loader.load_and_filter()
    assert len(df) == 18
    assert set(loader.label_map.keys()) == {"covid", "normal", "pneumonia"}
    assert loader.label_map["covid"] != loader.label_map["normal"]

    splits_df = loader.prepare_splits()
    assert "xray_split" in splits_df.columns
    assert (tmp_path / "label_map.json").is_file()
    assert (tmp_path / "splits.csv").is_file()


def test_model_architecture():
    model = ChestXrayClassifier(num_classes=5, backbone_name="resnet18", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 5)

    feats = model.extract_features(dummy_input)
    assert feats.dim() == 4


def test_dataset_builder_and_loader(tmp_path: Path):
    img_dir = tmp_path / "processed" / "chest_xray" / "normal"
    img_dir.mkdir(parents=True, exist_ok=True)
    img_path = img_dir / "test.png"
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(img_path)

    df = pd.DataFrame({
        "processed_image_path": [str(img_path)] * 4,
        "disease_label": [0, 1, 0, 1],
        "xray_split": ["train", "train", "val", "test"],
    })

    loaders = create_chest_xray_data_loaders(df, batch_size=2, num_workers=0)
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    for batch_imgs, batch_lbls in loaders["train"]:
        assert batch_imgs.shape == (2, 3, 224, 224)
        assert batch_lbls.shape == (2,)


def test_trainer_and_evaluator_dry_run(tmp_path: Path):
    df = pd.DataFrame({
        "processed_image_path": [str(tmp_path / "dummy.png")] * 6,
        "disease_label": [0, 1, 0, 1, 0, 1],
        "xray_split": ["train", "train", "val", "val", "test", "test"],
    })
    loaders = create_chest_xray_data_loaders(df, batch_size=2, num_workers=0)
    model = ChestXrayClassifier(num_classes=2, backbone_name="mobilenet_v3_small", pretrained=False)

    cfg = ChestXrayConfig().training
    cfg.epochs = 1
    cfg.use_tensorboard = False

    trainer = ChestXrayTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg,
        save_dir=tmp_path / "saved_models",
        label_map={"class_a": 0, "class_b": 1},
    )
    hist = trainer.fit()
    assert "train_loss" in hist
    assert len(hist["train_loss"]) == 1

    evaluator = ChestXrayEvaluator(
        model=model,
        test_loader=loaders["test"],
        label_map={"class_a": 0, "class_b": 1},
    )
    res = evaluator.evaluate()
    assert "global_metrics" in res
    assert "benchmark" in res


def test_predictor_standardized_json_and_clinical_report(tmp_path: Path):
    model = ChestXrayClassifier(num_classes=3, backbone_name="resnet18", pretrained=False)
    label_map = {"covid": 0, "normal": 1, "pneumonia": 2}
    predictor = ChestXrayPredictor(model=model, label_map=label_map)

    pil_img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    res = predictor.predict_image(pil_img)

    assert res.predicted_class in label_map
    assert 0.0 <= res.confidence <= 1.0
    assert len(res.top3_predictions) == 3

    std_dict = res.to_standard_dict()
    assert std_dict["module"] == "chest_xray"
    assert std_dict["modality"] == "chest_xray"
    assert "top3_predictions" in std_dict
    assert std_dict["status"] == "success"

    report, rpath = ClinicalReportGenerator.generate_report(res, output_path=tmp_path / "report.json")
    assert rpath.is_file()
    assert report["modality"] == "chest_xray"


def test_exporters(tmp_path: Path):
    model = ChestXrayClassifier(num_classes=2, backbone_name="resnet18", pretrained=False)
    label_map = {"normal": 0, "disease": 1}

    onnx_p = ChestXrayONNXExporter.export(model, label_map, export_path=tmp_path / "model.onnx")
    assert onnx_p.is_file()

    ts_p = TorchScriptExporter.export(model, export_path=tmp_path / "model.torchscript")
    assert ts_p.is_file()
