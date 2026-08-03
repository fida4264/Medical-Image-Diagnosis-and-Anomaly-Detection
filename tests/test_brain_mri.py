"""
tests/test_brain_mri.py
========================
Comprehensive pytest unit test suite for Module 7: Brain MRI Disease Analysis Framework.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from brain_mri.datasets.dataset_builder import BrainMRIDataset, create_brain_mri_data_loaders
from brain_mri.datasets.dataset_loader import BrainMRIDatasetLoader
from brain_mri.evaluation.evaluator import BrainMRIEvaluator
from brain_mri.export.onnx_export import BrainMRIONNXExporter
from brain_mri.export.torchscript_export import TorchScriptExporter
from brain_mri.inference.predict_brain_mri import BrainMRIPredictor
from brain_mri.models.brain_mri_model import BrainMRIClassifier
from brain_mri.reports.clinical_report import ClinicalReportGenerator
from brain_mri.training.trainer import BrainMRITrainer
from brain_mri.utils.checkpoint_manager import CheckpointManager
from brain_mri.utils.config import BrainMRIConfig, BrainMRITrainingConfig


@pytest.fixture
def dummy_brain_mri_df(tmp_path):
    """Creates dummy master metadata containing Brain MRI and other modalities."""
    rows = []
    # 4 Brain MRI classes
    classes = ["glioma", "meningioma", "notumor", "pituitary"]
    for cls_idx, cls_name in enumerate(classes):
        for i in range(10):
            rows.append({
                "image_id": f"brain_mri/{cls_name}_{i}.png",
                "internal_id": f"MRI_{cls_idx}_{i:03d}",
                "dataset_name": "Brain_MRI_DS",
                "dataset_source": "Kaggle",
                "modality": "brain_mri",
                "disease": cls_name,
                "normalized_class": cls_name,
                "original_class": cls_name,
                "split": "train" if i < 8 else "test",
                "image_path": str(tmp_path / "raw" / f"{cls_name}_{i}.png"),
                "processed_image_path": str(tmp_path / "processed" / "brain_mri" / cls_name / f"MRI_{cls_idx}_{i:03d}.png"),
                "extension": ".png",
                "width": 224,
                "height": 224,
                "channels": 3,
                "file_size_bytes": 2048,
                "is_valid": True,
                "md5_hash": f"hash_{cls_idx}_{i}",
            })

    # Add dummy chest_xray row to verify strict filtering
    rows.append({
        "image_id": "chest_xray/normal_1.png",
        "internal_id": "XRAY_001",
        "dataset_name": "XRay_DS",
        "dataset_source": "Kaggle",
        "modality": "chest_xray",
        "disease": "normal",
        "normalized_class": "normal",
        "original_class": "normal",
        "split": "train",
        "image_path": str(tmp_path / "raw" / "xray.png"),
        "processed_image_path": str(tmp_path / "processed" / "chest_xray" / "normal" / "XRAY_001.png"),
        "extension": ".png",
        "width": 224,
        "height": 224,
        "channels": 3,
        "file_size_bytes": 2048,
        "is_valid": True,
        "md5_hash": "hash_xray",
    })

    df = pd.DataFrame(rows)
    csv_path = tmp_path / "master_metadata.csv"
    df.to_csv(csv_path, index=False)
    return csv_path, df


def test_brain_mri_config():
    cfg = BrainMRIConfig.load()
    assert cfg.data.modality_filter == "brain_mri"
    assert cfg.model.backbone in ["efficientnet_b0", "resnet18", "resnet34", "mobilenet_v3_small", "densenet121", "convnext_tiny"]
    assert cfg.training.batch_size > 0


def test_dataset_loader_strict_filtering(dummy_brain_mri_df, tmp_path):
    csv_path, _ = dummy_brain_mri_df
    splits_path = tmp_path / "splits.csv"
    label_map_path = tmp_path / "label_map.json"

    loader = BrainMRIDatasetLoader(
        metadata_path=csv_path,
        processed_dir=tmp_path / "processed",
        splits_output_path=splits_path,
        label_map_output_path=label_map_path,
    )

    df = loader.load_and_filter()

    # Verify strictly 40 brain_mri records (chest_xray excluded)
    assert len(df) == 40
    assert set(df["modality"].unique()) == {"brain_mri"}
    assert len(loader.label_map) == 4

    splits_df = loader.prepare_splits()
    assert splits_path.is_file()
    assert label_map_path.is_file()
    assert set(splits_df["mri_split"].unique()) == {"train", "val", "test"}


def test_data_loader(dummy_brain_mri_df, tmp_path):
    csv_path, _ = dummy_brain_mri_df
    loader = BrainMRIDatasetLoader(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = loader.prepare_splits()

    loaders = create_brain_mri_data_loaders(splits_df, batch_size=4, target_size=(224, 224))
    assert "train" in loaders
    assert "val" in loaders
    assert "test" in loaders

    for imgs, labels in loaders["train"]:
        assert imgs.shape == (4, 3, 224, 224)
        assert labels.shape == (4,)
        break


def test_brain_mri_classifier_forward():
    model = BrainMRIClassifier(num_classes=4, backbone_name="efficientnet_b0", pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    logits = model(dummy_input)
    assert logits.shape == (2, 4)

    features = model.extract_features(dummy_input)
    assert features.dim() == 4


def test_checkpoint_manager(tmp_path):
    mgr = CheckpointManager(saved_model_dir=tmp_path)
    model = BrainMRIClassifier(num_classes=4, pretrained=False)
    optimizer = torch.optim.Adam(model.parameters())

    path = mgr.save_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=None,
        epoch=1,
        val_loss=0.5,
        val_f1=0.85,
        label_map={"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3},
        is_best=True,
    )

    assert path.is_file()
    assert mgr.best_path.is_file()

    epoch, val_loss, label_map = mgr.load_checkpoint(model=model, load_best=True)
    assert epoch == 1
    assert val_loss == 0.5
    assert len(label_map) == 4


def test_trainer_fit(dummy_brain_mri_df, tmp_path):
    csv_path, _ = dummy_brain_mri_df
    loader = BrainMRIDatasetLoader(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = loader.prepare_splits()
    loaders = create_brain_mri_data_loaders(splits_df, batch_size=8, target_size=(224, 224))

    model = BrainMRIClassifier(num_classes=4, backbone_name="mobilenet_v3_small", pretrained=False)
    cfg = BrainMRITrainingConfig(epochs=1, batch_size=8, use_tensorboard=False)

    save_dir = tmp_path / "saved_models"
    trainer = BrainMRITrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=cfg,
        save_dir=save_dir,
        label_map=loader.label_map,
        device="cpu",
    )

    history = trainer.fit()
    assert len(history["train_loss"]) == 1
    assert (save_dir / "last_model.pt").is_file()


def test_evaluator(dummy_brain_mri_df, tmp_path):
    csv_path, _ = dummy_brain_mri_df
    loader = BrainMRIDatasetLoader(
        metadata_path=csv_path,
        splits_output_path=tmp_path / "splits.csv",
        label_map_output_path=tmp_path / "label_map.json",
    )
    splits_df = loader.prepare_splits()
    loaders = create_brain_mri_data_loaders(splits_df, batch_size=8, target_size=(224, 224))

    model = BrainMRIClassifier(num_classes=4, backbone_name="mobilenet_v3_small", pretrained=False)
    evaluator = BrainMRIEvaluator(
        model=model,
        test_loader=loaders["test"],
        label_map=loader.label_map,
        device="cpu",
    )

    results = evaluator.evaluate()
    assert "global_metrics" in results
    assert "auc_scores" in results
    assert "benchmark" in results


def test_clinical_report_generator(tmp_path):
    gen = ClinicalReportGenerator(reports_dir=tmp_path)
    report = gen.generate_report(
        predicted_disease="glioma",
        confidence=0.95,
        probability_distribution={"glioma": 0.95, "meningioma": 0.03, "notumor": 0.01, "pituitary": 0.01},
        top_3_predictions=[{"disease": "glioma", "confidence": 0.95}],
        inference_time_ms=12.5,
    )

    assert report.predicted_disease == "glioma"
    assert (tmp_path / "brain_mri_report.json").is_file()


def test_predictor_and_export(tmp_path):
    model = BrainMRIClassifier(num_classes=4, backbone_name="mobilenet_v3_small", pretrained=False)
    ckpt_path = tmp_path / "best_model.pt"
    label_map_path = tmp_path / "label_map.json"

    label_map = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}
    with open(label_map_path, "w") as f:
        json.dump({"class_to_label": label_map, "label_to_class": {i: c for c, i in label_map.items()}}, f)

    torch.save({"model_state_dict": model.state_dict(), "backbone": "mobilenet_v3_small"}, ckpt_path)

    # Predictor
    predictor = BrainMRIPredictor(checkpoint_path=ckpt_path, label_map_path=label_map_path, device="cpu")
    dummy_img = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    res = predictor.predict(dummy_img)

    assert res.predicted_disease in label_map
    assert 0.0 <= res.confidence <= 1.0

    # ONNX Export
    onnx_exporter = BrainMRIONNXExporter(model=model)
    onnx_path = tmp_path / "best_model.onnx"
    exported_onnx = onnx_exporter.export(output_path=onnx_path)
    assert exported_onnx.is_file()
    assert onnx_exporter.verify(exported_onnx)

    # TorchScript Export
    ts_exporter = TorchScriptExporter(model=model)
    ts_path = tmp_path / "best_model.torchscript"
    exported_ts = ts_exporter.export(output_path=ts_path)
    assert exported_ts.is_file()
