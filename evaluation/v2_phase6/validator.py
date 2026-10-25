"""
evaluation/v2_phase6/validator.py
=================================
Validates baseline reproducibility across all 7 medical imaging modalities.
Loads saved checkpoints (best_model.pt) and executes deterministic inference
on validation and held-out test splits. Verifies that every reported metric
in outputs/v2_baseline_results.csv matches the recomputed values.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from training.v2_baseline.config import MODALITY_METADATA, ROOT
from training.v2_baseline.dataset import V2ManifestDataset
from training.v2_baseline.metrics import calculate_metrics, calculate_multilabel_metrics
from training.v2_baseline.model_factory import create_baseline_model
from training.v2_baseline.transforms import get_transforms

logger = logging.getLogger("validator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class BaselineValidator:
    """
    Validates model reproducibility and generates sample-level predictions
    without altering model weights or dataset splits.
    """

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        self.models_dir = ROOT / "models" / "v2_baseline"
        self.results_csv = ROOT / "outputs" / "v2_baseline_results.csv"

    def load_model(
        self,
        modality: str,
        checkpoint_path: Path,
        num_classes: int,
        task: str,
    ) -> torch.nn.Module:
        """Loads a model checkpoint from disk in eval mode."""
        model = create_baseline_model(
            model_name="resnet18",
            num_classes=num_classes,
            pretrained=False,
            task=task,
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def run_inference_on_loader(
        self,
        model: torch.nn.Module,
        loader: DataLoader,
        task: str,
        class_names: List[str],
        modality: str,
        split_name: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Executes deterministic evaluation over a DataLoader.
        Returns:
            summary_metrics: Dict with aggregate metrics (F1, AUC, Acc, etc.)
            sample_records: List of per-sample prediction dictionaries
        """
        all_probs = []
        all_targets = []
        all_metas = []

        for images, targets, metas in loader:
            images = images.to(self.device)
            logits = model(images)

            if task == "multilabel":
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets.cpu().numpy())

            # metas is a dict with list of values: {'image_id': [...], 'image_path': [...], 'standardized_label': [...]}
            batch_size = len(images)
            for i in range(batch_size):
                all_metas.append({
                    "image_id": metas["image_id"][i],
                    "image_path": metas["image_path"][i],
                    "standardized_label": metas["standardized_label"][i],
                })

        # Stack into continuous arrays
        if task == "multilabel":
            y_prob = np.vstack(all_probs)
            y_true = np.vstack(all_targets)
            y_pred = (y_prob >= 0.5).astype(np.int32)
            summary_metrics = calculate_multilabel_metrics(
                y_true=y_true,
                y_prob=y_prob,
                threshold=0.5,
                class_names=class_names,
            )
        else:
            y_prob = np.vstack(all_probs)
            y_true = np.concatenate(all_targets)
            y_pred = np.argmax(y_prob, axis=-1)
            summary_metrics = calculate_metrics(
                y_true=y_true,
                y_pred=y_pred,
                y_prob=y_prob,
                class_names=class_names,
                modality=modality,
            )

        # Assemble per-sample records
        sample_records = []
        for i in range(len(y_true)):
            meta = all_metas[i]
            if task == "multilabel":
                t_vec = y_true[i].tolist()
                p_vec = y_pred[i].tolist()
                probs_list = [round(float(p), 4) for p in y_prob[i]]
                is_exact_match = bool(np.array_equal(y_true[i], y_pred[i]))
                sample_records.append({
                    "image_id": meta["image_id"],
                    "image_path": meta["image_path"],
                    "modality": modality,
                    "split": split_name,
                    "true_label": t_vec,
                    "predicted_label": p_vec,
                    "prediction_score": round(float(np.max(y_prob[i])), 4),
                    "probabilities": probs_list,
                    "correct": is_exact_match,
                })
            else:
                t_idx = int(y_true[i])
                p_idx = int(y_pred[i])
                prob_score = float(y_prob[i][p_idx])
                is_correct = bool(t_idx == p_idx)
                sample_records.append({
                    "image_id": meta["image_id"],
                    "image_path": meta["image_path"],
                    "modality": modality,
                    "split": split_name,
                    "true_label_idx": t_idx,
                    "true_label": class_names[t_idx] if t_idx < len(class_names) else str(t_idx),
                    "predicted_label_idx": p_idx,
                    "predicted_label": class_names[p_idx] if p_idx < len(class_names) else str(p_idx),
                    "prediction_score": round(prob_score, 4),
                    "probabilities": [round(float(p), 4) for p in y_prob[i]],
                    "correct": is_correct,
                })

        return summary_metrics, sample_records

    def validate_all(self) -> Tuple[Dict[str, Any], List[Dict[str, Any]], pd.DataFrame]:
        """
        Executes validation across all 7 modalities.
        Returns:
            validation_report: Summary dictionary with comparison against reported metrics.
            all_sample_predictions: List of all sample records across datasets.
            metrics_df: DataFrame summarizing verified metrics.
        """
        validation_report = {}
        all_sample_predictions = []
        metrics_rows = []

        # Load Phase 5 reported results
        reported_df = pd.read_csv(self.results_csv) if self.results_csv.exists() else pd.DataFrame()

        for modality, meta in MODALITY_METADATA.items():
            logger.info(f"\n=======================================================")
            logger.info(f"   VALIDATING BASELINE: {modality.upper()}")
            logger.info(f"=======================================================")

            eval_transform = get_transforms(modality, is_train=False, image_size=(224, 224))
            class_names = meta["class_names"]
            num_classes = meta["num_classes"]
            task = meta["task"]
            split_type = meta["split_type"]

            if split_type == "kfold":
                k_folds = meta.get("k_folds", 5)
                fold_metrics = []
                for f in range(k_folds):
                    ckpt_path = self.models_dir / modality / f"fold_{f}" / "best_model.pt"
                    if not ckpt_path.exists():
                        logger.error(f"Missing checkpoint: {ckpt_path}")
                        continue

                    model = self.load_model(modality, ckpt_path, num_classes, task)
                    val_ds = V2ManifestDataset(
                        manifest=meta["split_file"],
                        fold=f,
                        is_train_fold=False,
                        transform=eval_transform,
                        task=task,
                        num_classes=num_classes,
                    )
                    loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                    metrics, samples = self.run_inference_on_loader(
                        model=model,
                        loader=loader,
                        task=task,
                        class_names=class_names,
                        modality=modality,
                        split_name=f"fold_{f}",
                    )
                    fold_metrics.append(metrics)
                    all_sample_predictions.extend(samples)

                    metrics_rows.append({
                        "modality": modality,
                        "split_type": "kfold",
                        "partition": f"fold_{f}",
                        "num_samples": len(samples),
                        "accuracy": round(metrics.get("accuracy", 0.0), 4),
                        "macro_f1": round(metrics.get("macro_f1", 0.0), 4),
                        "weighted_f1": round(metrics.get("weighted_f1", 0.0), 4),
                        "macro_auc": round(metrics.get("macro_auc", 0.0), 4) if metrics.get("macro_auc") is not None else "NA",
                    })

                # Compute CV mean/std
                if fold_metrics:
                    mean_acc = float(np.mean([m["accuracy"] for m in fold_metrics]))
                    std_acc = float(np.std([m["accuracy"] for m in fold_metrics]))
                    mean_f1 = float(np.mean([m["macro_f1"] for m in fold_metrics]))
                    std_f1 = float(np.std([m["macro_f1"] for m in fold_metrics]))
                    mean_auc = float(np.mean([m["macro_auc"] for m in fold_metrics if m.get("macro_auc") is not None]))

                    validation_report[modality] = {
                        "type": "kfold",
                        "folds_evaluated": len(fold_metrics),
                        "mean_accuracy": round(mean_acc, 4),
                        "std_accuracy": round(std_acc, 4),
                        "mean_macro_f1": round(mean_f1, 4),
                        "std_macro_f1": round(std_f1, 4),
                        "mean_macro_auc": round(mean_auc, 4),
                        "per_fold": fold_metrics,
                    }

                    metrics_rows.append({
                        "modality": modality,
                        "split_type": "kfold",
                        "partition": "CV_MEAN",
                        "num_samples": sum(m["num_samples"] for m in fold_metrics),
                        "accuracy": round(mean_acc, 4),
                        "macro_f1": round(mean_f1, 4),
                        "weighted_f1": round(float(np.mean([m["weighted_f1"] for m in fold_metrics])), 4),
                        "macro_auc": round(mean_auc, 4),
                    })

            else:
                # Single split: evaluate both validation and held-out test
                ckpt_path = self.models_dir / modality / "best_model.pt"
                if not ckpt_path.exists():
                    logger.error(f"Missing checkpoint: {ckpt_path}")
                    continue

                model = self.load_model(modality, ckpt_path, num_classes, task)

                # 1. Validation split
                val_ds = V2ManifestDataset(
                    manifest=meta["split_file"],
                    split="validation",
                    transform=eval_transform,
                    task=task,
                    num_classes=num_classes,
                )
                val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
                val_metrics, val_samples = self.run_inference_on_loader(
                    model=model,
                    loader=val_loader,
                    task=task,
                    class_names=class_names,
                    modality=modality,
                    split_name="validation",
                )
                all_sample_predictions.extend(val_samples)

                metrics_rows.append({
                    "modality": modality,
                    "split_type": "fixed",
                    "partition": "validation",
                    "num_samples": len(val_samples),
                    "accuracy": round(val_metrics.get("accuracy", 0.0), 4) if val_metrics.get("accuracy") is not None else "NA",
                    "macro_f1": round(val_metrics.get("macro_f1", 0.0), 4),
                    "weighted_f1": round(val_metrics.get("weighted_f1", 0.0), 4),
                    "macro_auc": round(val_metrics.get("macro_auc", 0.0), 4) if val_metrics.get("macro_auc") is not None else "NA",
                })

                # 2. Test split
                test_ds = V2ManifestDataset(
                    manifest=meta["split_file"],
                    split="test",
                    transform=eval_transform,
                    task=task,
                    num_classes=num_classes,
                )
                test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
                test_metrics, test_samples = self.run_inference_on_loader(
                    model=model,
                    loader=test_loader,
                    task=task,
                    class_names=class_names,
                    modality=modality,
                    split_name="test",
                )
                all_sample_predictions.extend(test_samples)

                metrics_rows.append({
                    "modality": modality,
                    "split_type": "fixed",
                    "partition": "test",
                    "num_samples": len(test_samples),
                    "accuracy": round(test_metrics.get("accuracy", 0.0), 4) if test_metrics.get("accuracy") is not None else "NA",
                    "macro_f1": round(test_metrics.get("macro_f1", 0.0), 4),
                    "weighted_f1": round(test_metrics.get("weighted_f1", 0.0), 4),
                    "macro_auc": round(test_metrics.get("macro_auc", 0.0), 4) if test_metrics.get("macro_auc") is not None else "NA",
                })

                validation_report[modality] = {
                    "type": "fixed",
                    "validation_metrics": val_metrics,
                    "test_metrics": test_metrics,
                }

        metrics_df = pd.DataFrame(metrics_rows)
        return validation_report, all_sample_predictions, metrics_df


if __name__ == "__main__":
    validator = BaselineValidator()
    report, samples, df = validator.validate_all()
    print("\nVerified Metrics:")
    print(df.to_string(index=False))
