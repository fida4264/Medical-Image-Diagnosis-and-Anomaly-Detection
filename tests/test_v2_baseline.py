"""
tests/test_v2_baseline.py
=========================
Comprehensive 18-point automated test suite for V2.0 baseline model training & experiment framework.
Verifies dataset manifests, leakage guards, architectures, transforms, losses, metrics, and reproducibility.
"""

import ast
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms as T

from training.v2_baseline.config import ExperimentConfig, MODALITY_METADATA
from training.v2_baseline.dataset import V2ManifestDataset
from training.v2_baseline.transforms import get_transforms
from training.v2_baseline.model_factory import create_baseline_model
from training.v2_baseline.loss_factory import create_loss_function, compute_class_weights, compute_multilabel_pos_weights
from training.v2_baseline.metrics import calculate_metrics, calculate_multilabel_metrics
from training.v2_baseline.trainer import V2Trainer, set_seed
from training.v2_baseline.smoke_test import run_modality_smoke_test

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS_DIR = ROOT / "data" / "manifests"
SPLITS_DIR = MANIFESTS_DIR / "splits"


# Check 1: Correct manifest loading
def test_01_manifest_loading():
    for mod, meta in MODALITY_METADATA.items():
        p = Path(meta["split_file"])
        assert p.exists(), f"Split manifest missing for {mod}: {p}"
        df = pd.read_csv(p)
        assert len(df) > 0, f"Empty manifest for {mod}"
        assert "image_id" in df.columns
        assert "image_path" in df.columns


# Check 2: Correct class mapping
def test_02_class_mapping():
    for mod, meta in MODALITY_METADATA.items():
        df = pd.read_csv(meta["split_file"])
        if meta["task"] != "multilabel":
            assert "class_id" in df.columns
            unique_ids = sorted(df["class_id"].unique())
            expected_ids = list(range(meta["num_classes"]))
            assert unique_ids == expected_ids, f"Class ID mismatch for {mod}: {unique_ids} vs {expected_ids}"


# Check 3: No quarantine records loaded
def test_03_no_quarantine_records_loaded():
    quar_path = MANIFESTS_DIR / "v2_quarantine.csv"
    assert quar_path.exists()
    quar_df = pd.read_csv(quar_path)
    quar_ids = set(quar_df["image_id"].astype(str))

    # Passing quarantine df to dataset should be rejected defensively
    with pytest.raises(ValueError):
        V2ManifestDataset(quar_df)

    # Split manifests must contain 0 quarantine IDs
    for mod, meta in MODALITY_METADATA.items():
        df = pd.read_csv(meta["split_file"])
        split_ids = set(df["image_id"].astype(str))
        assert len(split_ids.intersection(quar_ids)) == 0, f"Quarantine contamination in {mod}!"


# Check 4: No mask records loaded
def test_04_no_mask_records_loaded():
    for mod, meta in MODALITY_METADATA.items():
        df = pd.read_csv(meta["split_file"])
        for path in df["image_path"]:
            assert "_mask" not in path.lower() or "mask" not in Path(path).stem.lower(), f"Mask found: {path}"


# Check 5: No VinBigData records loaded
def test_05_no_vinbigdata_records_loaded():
    cxr_df = pd.read_csv(MODALITY_METADATA["chest_xray"]["split_file"])
    if "dataset_name" in cxr_df.columns:
        assert not (cxr_df["dataset_name"] == "VinBigData Chest X-ray Abnormalities Detection").any()
    for path in cxr_df["image_path"]:
        assert "vinbigdata" not in path.lower()


# Check 6: Retina uses multi-label targets
def test_06_retina_multilabel_targets():
    df = pd.read_csv(MODALITY_METADATA["retina"]["split_file"])
    assert "multi_label_vector" in df.columns
    sample_vec_str = df["multi_label_vector"].iloc[0]
    vec = ast.literal_eval(sample_vec_str)
    assert isinstance(vec, list)
    assert len(vec) == 8
    assert all(v in [0, 1] for v in vec)


# Check 7: Retina output dimension = 8
def test_07_retina_output_dimension():
    model = create_baseline_model(model_name="resnet18", num_classes=8, pretrained=False, task="multilabel")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 8)


# Check 8: Skin output dimension = 7
def test_08_skin_output_dimension():
    model = create_baseline_model(model_name="resnet18", num_classes=7, pretrained=False, task="multiclass")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    assert out.shape == (2, 7)


# Check 9: Correct loss function
def test_09_correct_loss_function():
    # Multilabel must use BCEWithLogitsLoss
    loss_retina = create_loss_function(loss_type="bce_with_logits", task="multilabel", num_classes=8)
    assert isinstance(loss_retina, torch.nn.BCEWithLogitsLoss)

    # Multiclass must use CrossEntropyLoss
    loss_mri = create_loss_function(loss_type="cross_entropy", task="multiclass", num_classes=4)
    assert isinstance(loss_mri, torch.nn.CrossEntropyLoss)


# Check 10: Validation has no random augmentation
def test_10_validation_no_random_augmentation():
    val_t = get_transforms("skin", is_train=False)
    for t in val_t.transforms:
        assert not isinstance(t, (T.RandomHorizontalFlip, T.RandomAffine, T.ColorJitter, T.RandomRotation))


# Check 11: Test has no random augmentation
def test_11_test_no_random_augmentation():
    test_t = get_transforms("brain_mri", is_train=False)
    for t in test_t.transforms:
        assert not isinstance(t, (T.RandomHorizontalFlip, T.RandomAffine, T.ColorJitter, T.RandomRotation))


# Check 12: Checkpoint metadata exists
def test_12_checkpoint_metadata_exists():
    with tempfile.TemporaryDirectory() as tmpdir:
        config = ExperimentConfig.from_modality(
            "brain_ct",
            checkpoint_dir=tmpdir,
            output_dir=tmpdir,
            epochs=1,
            batch_size=4,
            fold=0,
        )
        trainer = V2Trainer(config)
        trainer._save_checkpoint("test_ckpt.pt", epoch=1, metrics={"accuracy": 0.8}, is_best=True)
        ckpt_file = Path(tmpdir) / "brain_ct" / "fold_0" / "test_ckpt.pt"
        assert ckpt_file.exists()
        loaded = torch.load(ckpt_file, map_location="cpu")
        assert "metadata" in loaded
        assert loaded["metadata"]["modality"] == "brain_ct"
        assert loaded["metadata"]["seed"] == 42


# Check 13: Reproducibility configuration exists
def test_13_reproducibility_configuration_exists():
    config = ExperimentConfig.from_modality("brain_mri")
    assert config.seed == 42
    d = config.to_dict()
    assert d["seed"] == 42
    assert d["modality"] == "brain_mri"


# Check 14: Training can complete one miniature smoke-test epoch
def test_14_smoke_test_epoch():
    res = run_modality_smoke_test("brain_ct", num_samples=16)
    assert res["status"] == "PASSED"
    assert "loss" in res


# Check 15: Metrics calculation works
def test_15_metrics_calculation_works():
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 2, 0, 1, 2, 3])
    metrics = calculate_metrics(y_true, y_pred, class_names=["c0", "c1", "c2", "c3"])
    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert "per_class" in metrics
    assert metrics["accuracy"] == 7 / 8


# Check 16: Confusion matrix generation works
def test_16_confusion_matrix_generation():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = calculate_metrics(y_true, y_pred, class_names=["norm", "hem"])
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2
    assert len(cm[0]) == 2


# Check 17: Multi-label metric calculation works
def test_17_multilabel_metrics_calculation():
    y_true = np.array([[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 1]])
    y_prob = np.array([[0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [0.1, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.7]])
    metrics = calculate_multilabel_metrics(y_true, y_prob)
    assert "macro_f1" in metrics
    assert "micro_f1" in metrics
    assert "per_label" in metrics
    assert metrics["macro_f1"] > 0.0


# Check 18: CV fold handling works
def test_18_cv_fold_handling():
    bct_meta = MODALITY_METADATA["brain_ct"]
    train_ds = V2ManifestDataset(bct_meta["split_file"], fold=0, is_train_fold=True, task="binary")
    val_ds = V2ManifestDataset(bct_meta["split_file"], fold=0, is_train_fold=False, task="binary")
    assert len(train_ds) == 160  # 4 folds * 40
    assert len(val_ds) == 40    # 1 fold * 40
    # No overlap
    train_ids = set(train_ds.df["image_id"])
    val_ids = set(val_ds.df["image_id"])
    assert len(train_ids.intersection(val_ids)) == 0
