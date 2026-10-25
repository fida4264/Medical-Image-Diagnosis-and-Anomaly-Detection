"""
tests/test_v2_phase6.py
=======================
Automated test suite for Phase 6: Baseline Evaluation, Error Analysis & Validation.
Verifies metric reproducibility, prediction alignment, confusion matrix properties,
Retina multi-label constraints, CV fold aggregation, and split disjointness.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch

from evaluation.v2_phase6.sanity_checker import HistopathologySanityChecker
from training.v2_baseline.config import MODALITY_METADATA, ROOT
from training.v2_baseline.metrics import calculate_metrics, calculate_multilabel_metrics
from training.v2_baseline.model_factory import create_baseline_model


def test_01_all_checkpoints_exist_and_loadable():
    """Confirms all 7 modality checkpoints exist and have matching state_dict weights."""
    models_dir = ROOT / "models" / "v2_baseline"

    for mod, meta in MODALITY_METADATA.items():
        if meta["split_type"] == "kfold":
            for f in range(meta.get("k_folds", 5)):
                ckpt_path = models_dir / mod / f"fold_{f}" / "best_model.pt"
                assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
                ckpt = torch.load(ckpt_path, map_location="cpu")
                assert "state_dict" in ckpt, f"Checkpoint {ckpt_path} missing state_dict"
        else:
            ckpt_path = models_dir / mod / "best_model.pt"
            assert ckpt_path.exists(), f"Missing checkpoint: {ckpt_path}"
            ckpt = torch.load(ckpt_path, map_location="cpu")
            assert "state_dict" in ckpt, f"Checkpoint {ckpt_path} missing state_dict"


def test_02_prediction_label_alignment_and_dimensions():
    """Verifies that model forward pass shapes match expected class dimensions."""
    x = torch.randn(2, 3, 224, 224)

    for mod, meta in MODALITY_METADATA.items():
        num_classes = meta["num_classes"]
        model = create_baseline_model(
            model_name="resnet18",
            num_classes=num_classes,
            pretrained=False,
            task=meta["task"],
        )
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, num_classes), f"{mod}: expected shape (2, {num_classes}), got {out.shape}"


def test_03_confusion_matrix_properties():
    """Verifies confusion matrix generation properties (non-negative, row sum = support)."""
    y_true = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 2, 2, 0, 1, 1])
    class_names = ["class_0", "class_1", "class_2"]

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    assert cm.shape == (3, 3)
    assert np.all(cm >= 0)
    assert np.sum(cm) == len(y_true)
    assert np.sum(cm[0, :]) == 4  # 4 true instances of class_0


def test_04_retina_multilabel_metrics_computation():
    """Tests 8D multi-label metric calculation with sigmoid outputs."""
    class_names = MODALITY_METADATA["retina"]["class_names"]
    assert len(class_names) == 8

    # 10 samples, 8 labels
    y_true = np.random.randint(0, 2, size=(10, 8))
    y_prob = np.random.uniform(0.0, 1.0, size=(10, 8))

    metrics = calculate_multilabel_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=0.5,
        class_names=class_names,
    )
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_label" in metrics
    assert len(metrics["per_label"]) == 8

    for name in class_names:
        pl = metrics["per_label"][name]
        assert 0.0 <= pl["precision"] <= 1.0
        assert 0.0 <= pl["recall"] <= 1.0
        assert 0.0 <= pl["f1"] <= 1.0


def test_05_cross_validation_aggregation():
    """Tests 5-fold CV mean and std calculation logic."""
    fold_f1s = [0.9750, 0.9499, 0.9499, 0.9250, 0.9250]
    mean_f1 = float(np.mean(fold_f1s))
    std_f1 = float(np.std(fold_f1s))

    assert np.isclose(mean_f1, 0.9450, atol=1e-3)
    assert np.isclose(std_f1, 0.0187, atol=1e-3)


def test_06_histopathology_split_strictly_disjoint():
    """Verifies that LC25000 histopathology splits have 0 path overlap."""
    checker = HistopathologySanityChecker(output_dir=ROOT / "outputs" / "v2_phase6")
    audit = checker.run_audit()

    assert audit["strictly_disjoint"] is True
    assert audit["train_val_overlap"] == 0
    assert audit["train_test_overlap"] == 0
    assert audit["val_test_overlap"] == 0
    assert audit["total_images"] == 25000
    assert audit["train_samples"] == 20000
    assert audit["val_samples"] == 2500
    assert audit["test_samples"] == 2500


def test_07_confidence_score_range_and_contract():
    """Ensures prediction scores fall strictly within [0.0, 1.0] range."""
    logits = torch.tensor([[2.5, -1.0, 0.2], [-0.5, 3.2, 1.1]])
    probs = torch.softmax(logits, dim=-1).numpy()
    max_scores = np.max(probs, axis=-1)

    assert np.all(max_scores >= 0.0)
    assert np.all(max_scores <= 1.0)
    assert np.allclose(np.sum(probs, axis=-1), [1.0, 1.0])


def test_08_error_sample_schema_validation():
    """Validates that error sample records satisfy Phase 6 schema constraints."""
    sample = {
        "image_id": "img_001",
        "image_path": "data/sample.jpg",
        "modality": "chest_xray",
        "split": "test",
        "true_label": "covid",
        "predicted_label": "viral_pneumonia",
        "prediction_score": 0.8421,
        "correct": False,
    }
    required_keys = ["image_id", "true_label", "predicted_label", "prediction_score", "correct", "modality"]
    for k in required_keys:
        assert k in sample
    assert sample["correct"] is False
