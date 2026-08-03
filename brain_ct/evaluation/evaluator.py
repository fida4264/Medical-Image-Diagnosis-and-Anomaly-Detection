"""
brain_ct/evaluation/evaluator.py
=================================
Full test-set evaluation for Brain CT Disease Model.
Generates confusion matrix, ROC curves, misclassified grids, benchmark results, and reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from brain_ct.evaluation.benchmark import benchmark_inference
from brain_ct.evaluation.confusion_matrix import plot_confusion_matrix
from brain_ct.evaluation.misclassified import extract_misclassified_samples, plot_misclassified_grid
from brain_ct.evaluation.roc import plot_roc_curves
from brain_ct.training.metrics import compute_weighted_f1
from brain_ct.utils.config import BrainCTEvaluationConfig
from modality_classifier.training.metrics import compute_classification_metrics  # REUSE Module 6

logger = logging.getLogger(__name__)


class BrainCTEvaluator:
    """Complete evaluation engine for Brain CT Disease Classification model."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_map: Dict[str, int],
        config: Optional[BrainCTEvaluationConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.inv_label_map = {idx: name for name, idx in label_map.items()}
        self.config = config or BrainCTEvaluationConfig()
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device).eval()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        all_probs, all_targets = [], []

        for images, targets in self.test_loader:
            images = images.to(self.device)
            probs = torch.softmax(self.model(images), dim=1).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

        y_probs = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.argmax(y_probs, axis=1)

        global_metrics = compute_classification_metrics(y_true, y_probs)
        global_metrics["f1_weighted"] = compute_weighted_f1(y_true, y_pred)

        class_names = [self.inv_label_map.get(i, f"Class_{i}") for i in range(len(self.label_map))]
        per_class_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        auc_scores = plot_roc_curves(y_true, y_probs, class_names, save_dir=self.config.roc_curves_dir)

        dataset = getattr(self.test_loader, "dataset", None)
        if dataset and hasattr(dataset, "df") and "processed_image_path" in dataset.df.columns:
            all_paths = dataset.df["processed_image_path"].tolist()
        else:
            all_paths = [f"Sample_{i}" for i in range(len(y_true))]

        misclassified = extract_misclassified_samples(y_true, y_pred, y_probs, all_paths, class_names)
        plot_misclassified_grid(misclassified, save_path=Path(self.config.misclassified_dir) / "grid.png")

        benchmark = benchmark_inference(self.model, device=self.device.type)

        return {
            "global_metrics": global_metrics,
            "per_class_report": per_class_report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": (cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)).tolist(),
            "auc_scores": auc_scores,
            "class_names": class_names,
            "misclassified_count": len(misclassified),
            "benchmark": benchmark,
            "total_test_samples": len(y_true),
        }

    def save_reports(self, results: Dict[str, Any]) -> Tuple[Path, Path]:
        json_path = Path(self.config.classification_report_json)
        md_path = Path(self.config.performance_report_md)
        cm_path = Path(self.config.confusion_matrix_png)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        plot_confusion_matrix(
            np.array(results["confusion_matrix"]),
            results["class_names"],
            cm_path,
            title="Brain CT Confusion Matrix",
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        gm = results["global_metrics"]
        bm = results["benchmark"]
        md_lines = [
            "# Module 9: Brain CT Disease Analysis Performance Report",
            "",
            "## 1. Global Classification Performance",
            "",
            f"- **Total Test Samples:** {results['total_test_samples']:,}",
            f"- **Accuracy:** {gm['accuracy']*100:.2f}%",
            f"- **Macro Precision:** {gm['precision_macro']:.4f}",
            f"- **Macro Recall:** {gm['recall_macro']:.4f}",
            f"- **Macro F1:** {gm['f1_macro']:.4f}",
            f"- **Weighted F1:** {gm['f1_weighted']:.4f}",
            "",
            "## 2. Per-Disease Performance & AUC",
            "",
            "| Disease Class | Precision | Recall | F1-Score | Support | ROC AUC |",
            "|---|---|---|---|---|---|",
        ]

        for cls_name in results["class_names"]:
            if cls_name in results["per_class_report"]:
                c = results["per_class_report"][cls_name]
                auc_val = results["auc_scores"].get(cls_name, 0.0)
                md_lines.append(
                    f"| `{cls_name}` | {c['precision']:.4f} | {c['recall']:.4f} | "
                    f"{c['f1-score']:.4f} | {int(c['support']):,} | {auc_val:.4f} |"
                )

        md_lines.extend([
            "",
            "## 3. Inference Benchmark",
            "",
            f"- **Device:** `{bm['device']}`",
            f"- **Mean Latency:** {bm['mean_latency_ms']} ms",
            f"- **p50 Latency:** {bm['p50_latency_ms']} ms",
            f"- **p95 Latency:** {bm['p95_latency_ms']} ms",
            f"- **Throughput:** {bm['throughput_fps']} FPS",
            f"- **Misclassified:** {results['misclassified_count']}",
            "",
            "---",
            "*Report generated automatically by Module 9 Evaluation Engine.*",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved reports to {json_path} and {md_path}")
        return json_path, md_path
