"""
training/v2_baseline/smoke_test.py
==================================
Miniature smoke test pipeline for rapid end-to-end verification of all 7 modalities.
Runs 1 mini-epoch on a small subset (16-32 samples) per modality before full training.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .config import ExperimentConfig, MODALITY_METADATA
from .dataset import V2ManifestDataset
from .transforms import get_transforms
from .model_factory import create_baseline_model
from .loss_factory import create_loss_function
from .metrics import calculate_metrics, calculate_multilabel_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("smoke_test")


def run_modality_smoke_test(modality: str, num_samples: int = 16) -> Dict[str, Any]:
    """
    Executes a 1-step end-to-end pipeline smoke test for a single modality.
    """
    logger.info(f"==> Running smoke test for modality: {modality}...")
    meta = MODALITY_METADATA[modality]
    task = meta["task"]
    num_classes = meta["num_classes"]
    split_file = meta["split_file"]

    # 1. Load manifest and take a small subset
    df = pd.read_csv(split_file)
    if "split" in df.columns:
        subset_df = df[df["split"] == "train"].head(num_samples).reset_index(drop=True)
    elif "fold" in df.columns:
        subset_df = df[df["fold"] != 0].head(num_samples).reset_index(drop=True)
    else:
        subset_df = df.head(num_samples).reset_index(drop=True)

    # 2. Dataset and Loader
    train_transform = get_transforms(modality, is_train=True, image_size=(224, 224))
    dataset = V2ManifestDataset(
        manifest=subset_df,
        transform=train_transform,
        task=task,
        num_classes=num_classes,
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 3. Model
    model = create_baseline_model(
        model_name="resnet18",
        num_classes=num_classes,
        pretrained=False,  # faster init for smoke test
        task=task,
    )
    model.train()

    # 4. Criterion & Optimizer
    criterion = create_loss_function(
        loss_type="bce_with_logits" if task == "multilabel" else "cross_entropy",
        task=task,
        num_classes=num_classes,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # 5. One forward + backward pass
    batch_images, batch_targets, _ = next(iter(loader))
    optimizer.zero_grad()
    logits = model(batch_images)

    expected_shape = (batch_images.shape[0], num_classes)
    assert logits.shape == expected_shape, f"Logit shape mismatch: {logits.shape} vs {expected_shape}"

    loss = criterion(logits, batch_targets)
    assert not torch.isnan(loss), "Loss is NaN!"
    loss.backward()
    optimizer.step()

    # 6. Evaluation metrics test
    model.eval()
    with torch.no_grad():
        eval_logits = model(batch_images)
        if task == "multilabel":
            eval_probs = torch.sigmoid(eval_logits).cpu().numpy()
            metrics = calculate_multilabel_metrics(
                y_true=batch_targets.cpu().numpy(),
                y_prob=eval_probs,
                class_names=meta["class_names"],
            )
        else:
            eval_probs = torch.softmax(eval_logits, dim=-1).cpu().numpy()
            eval_preds = eval_probs.argmax(axis=-1)
            metrics = calculate_metrics(
                y_true=batch_targets.cpu().numpy(),
                y_pred=eval_preds,
                y_prob=eval_probs,
                class_names=meta["class_names"],
                modality=modality,
            )

    logger.info(f"✅ Smoke test PASSED for {modality}. Initial Loss: {loss.item():.4f}")
    return {
        "modality": modality,
        "status": "PASSED",
        "output_shape": list(logits.shape),
        "loss": float(loss.item()),
        "metrics_keys": list(metrics.keys()),
    }


def run_all_smoke_tests() -> Dict[str, Any]:
    """Runs smoke tests across all 7 modalities."""
    results = {}
    all_passed = True

    for modality in MODALITY_METADATA.keys():
        try:
            res = run_modality_smoke_test(modality)
            results[modality] = res
        except Exception as e:
            logger.error(f"❌ Smoke test FAILED for {modality}: {e}", exc_info=True)
            results[modality] = {"status": "FAILED", "error": str(e)}
            all_passed = False

    return {
        "all_passed": all_passed,
        "results": results,
    }


if __name__ == "__main__":
    report = run_all_smoke_tests()
    if report["all_passed"]:
        print("\nAll 7 modality smoke tests passed successfully!")
        sys.exit(0)
    else:
        print("\nSome smoke tests failed.")
        sys.exit(1)
