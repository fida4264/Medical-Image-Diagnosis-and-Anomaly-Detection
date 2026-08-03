"""
tests/test_modality_classifier.py
==================================
Comprehensive pytest unit test suite for Module 6: Modality Classifier.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from modality_classifier.config import ModalityClassifierConfig, DataConfig, ModelConfig, TrainingConfig
from modality_classifier.datasets.data_loader import MedicalModalityDataset, create_data_loaders
from modality_classifier.datasets.dataset_builder import ModalityDatasetBuilder
from modality_classifier.evaluation.evaluator import ModalityEvaluator
from modality_classifier.inference.onnx_exporter import ONNXExporter
from modality_classifier.inference.predictor import ModalityPredictor
from modality_classifier.models.backbone_factory import BackboneFactory
from modality_classifier.models.base_model import ModalityClassifier, build_modality_classifier
from modality_classifier.models.classifier_head import ModularClassifierHead
from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint
from modality_classifier.training.metrics import compute_classification_metrics
from modality_classifier.training.trainer import ModalityTrainer


@pytest.fixture
def dummy_metadata_df(tmp_path):
    """Creates dummy metadata DataFrame with 7 modalities for testing."""
    modalities = ["brain_mri", "brain_ct", "chest_xray", "histopathology", "retina", "skin", "ultrasound"]
    rows = []
    for mod_idx, mod in enumerate(modalities):
        for i in range(10):
            img_name = f"{mod}_{i:03d}"
            rows.append({
                "image_id": f"{mod}/{img_name}.png",
                "internal_id": f"ID_{mod_idx}_{i}",
                "dataset_name": f"{mod}_ds",
                "dataset_source": "test",
                "modality": mod,
                "disease": "normal",
                "normalized_class": "normal",
                "original_class": "normal",
                "split": "unspecified",
                "image_path": str(tmp_path / "raw" / f"{img_name}.png"),
                "processed_image_path": str(tmp_path / "processed" / mod / "normal" / f"{img_name}.png"),
                "extension": ".png",
                "width": 224,
                "height": 224,
                "channels": 3,
                "file_size_bytes": 2048,
                "is_valid": True,
                "md5_hash": f"hash_{mod_idx}_{i}",
            })
    df = pd.DataFrame(rows)
    csv_path = tmp_path / "master_metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_config_loader(tmp_path):
    cfg = ModalityClassifierConfig.load()
    assert cfg.data.target_size == (224, 224)
    assert cfg.model.backbone in ["mobilenet_v3_small", "resnet18", "resnet34", "efficientnet_b0"]
    assert cfg.training.batch_size > 0


def test_dataset_builder(dummy_metadata_df, tmp_path):
    csv_path, _ = dummy_metadata_df
    splits_path = tmp_path / "splits.csv"
    label_map_path = tmp_path / "label_map.json"

    builder = ModalityDatasetBuilder(
        metadata_path=csv_path,
        processed_dir=tmp_path / "processed",
        splits_output_path=splits_path,
        label_map_output_path=label_map_path,
    )

    df = builder.load_and_prepare()
    assert len(builder.label_map) == 7
    assert "modality_label" in df.columns

    splits_df = builder.create_splits()
    assert splits_path.is_file()
    assert label_map_path.is_file()
    assert set(splits_df["modality_split"].unique()) == {"train", "val", "test"}


def test_data_loader(dummy_metadata_df, tmp_path):
    csv_path, _ = dummy_metadata_df
    builder = ModalityDatasetBuilder(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = builder.create_splits()

    loaders = create_data_loaders(splits_df, batch_size=4, target_size=(224, 224))
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    for imgs, labels in loaders["train"]:
        assert imgs.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        break


def test_backbone_factory():
    for name in ["mobilenet_v3_small", "resnet18"]:
        backbone, out_feats = BackboneFactory.create_backbone(name=name, pretrained=False)
        assert isinstance(backbone, torch.nn.Module)
        assert out_feats > 0


def test_modality_classifier_forward():
    model = ModalityClassifier(num_classes=7, backbone_name="mobilenet_v3_small", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 7)

    features = model.extract_features(dummy_input)
    assert features.dim() == 4


def test_classification_metrics():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6])
    y_probs = np.eye(7)
    metrics = compute_classification_metrics(y_true, y_probs)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1_macro"] == 1.0
    assert metrics["top_3_accuracy"] == 1.0


def test_early_stopping_and_checkpoint(tmp_path):
    es = EarlyStopping(patience=2, min_delta=1e-3, mode="min")
    assert not es(1.0)
    assert not es(0.9)
    assert not es(0.8999)  # no improvement
    assert es(0.8999)  # patience reached -> stop

    ckpt_path = tmp_path / "best_model.pt"
    ckpt = ModelCheckpoint(filepath=ckpt_path, mode="min")
    model = ModalityClassifier(num_classes=7, pretrained=False)

    saved = ckpt.step(0.5, model, metadata={"epoch": 1})
    assert saved
    assert ckpt_path.is_file()


def test_trainer_fit(dummy_metadata_df, tmp_path):
    csv_path, _ = dummy_metadata_df
    builder = ModalityDatasetBuilder(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = builder.create_splits()
    loaders = create_data_loaders(splits_df, batch_size=8, target_size=(224, 224))

    model = ModalityClassifier(num_classes=7, backbone_name="mobilenet_v3_small", pretrained=False)
    cfg = TrainingConfig(epochs=1, batch_size=8)

    save_path = tmp_path / "model_best.pt"
    trainer = ModalityTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg,
        save_path=save_path,
        label_map=builder.label_map,
        device="cpu",
    )

    history = trainer.fit()
    assert len(history["train_loss"]) == 1
    assert save_path.is_file()


def test_evaluator(dummy_metadata_df, tmp_path):
    csv_path, _ = dummy_metadata_df
    builder = ModalityDatasetBuilder(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = builder.create_splits()
    loaders = create_data_loaders(splits_df, batch_size=8, target_size=(224, 224))

    model = ModalityClassifier(num_classes=7, backbone_name="mobilenet_v3_small", pretrained=False)
    evaluator = ModalityEvaluator(
        model=model,
        test_loader=loaders["test"],
        label_map=builder.label_map,
        device="cpu",
    )

    results = evaluator.evaluate()
    assert "global_metrics" in results
    assert "confusion_matrix" in results


def test_predictor_and_onnx(tmp_path):
    model = ModalityClassifier(num_classes=7, backbone_name="mobilenet_v3_small", pretrained=False)
    save_path = tmp_path / "model_best.pt"
    label_map_path = tmp_path / "label_map.json"

    label_map = {m: i for i, m in enumerate(["brain_mri", "brain_ct", "chest_xray", "histopathology", "retina", "skin", "ultrasound"])}
    with open(label_map_path, "w") as f:
        json.dump({"modality_to_label": label_map, "label_to_modality": {i: m for m, i in label_map.items()}}, f)

    torch.save({"model_state_dict": model.state_dict(), "metadata": {"backbone": "mobilenet_v3_small"}}, save_path)

    # Predictor
    predictor = ModalityPredictor(checkpoint_path=save_path, label_map_path=label_map_path, device="cpu")
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)

    assert res.predicted_modality in label_map
    assert 0.0 <= res.confidence <= 1.0
    assert "Module" in res.routing_destination

    # ONNX Exporter
    exporter = ONNXExporter(model=model)
    onnx_path = tmp_path / "model.onnx"
    exported_path = exporter.export(output_path=onnx_path)
    assert exported_path.is_file()

    verified = exporter.verify(exported_path)
    assert verified
