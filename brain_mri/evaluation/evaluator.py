"""
brain_mri/evaluation/evaluator.py
==================================
Evaluates Brain MRI model on test set, generates confusion matrix, ROC curves, misclassified grids, and reports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from brain_mri.evaluation.benchmark import benchmark_inference
from brain_mri.evaluation.misclassified import extract_misclassified_samples, plot_misclassified_grid
from brain_mri.evaluation.roc import plot_roc_curves
from brain_mri.training.metrics import compute_weighted_f1
from brain_mri.utils.config import BrainMRIEvaluationConfig
from modality_classifier.training.metrics import compute_classification_metrics  # REUSE Module 6

logger = logging.getLogger(__name__)


class BrainMRIEvaluator:
    """
    Complete evaluation engine for Brain MRI Disease Classification model.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_map: Dict[str, int],
        config: Optional[BrainMRIEvaluationConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.inv_label_map = {idx: name for name, idx in label_map.items()}
        self.config = config or BrainMRIEvaluationConfig()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Runs evaluation on test set and returns full metrics dictionary.
        """
        all_probs = []
        all_targets = []
        all_paths = []

        for batch in self.test_loader:
            images, targets = batch[0], batch[1]
            images = images.to(self.device)
            outputs = self.model(images)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()

            all_probs.append(probs)
            all_targets.append(targets.numpy())

        y_probs = np.concatenate(all_probs, axis=0)
        y_true = np.concatenate(all_targets, axis=0)
        y_pred = np.argmax(y_probs, axis=1)

        # Global metrics
        global_metrics = compute_classification_metrics(y_true, y_probs)
        global_metrics["f1_weighted"] = compute_weighted_f1(y_true, y_pred)

        # Per-class report
        class_names = [self.inv_label_map.get(i, f"Class_{i}") for i in range(len(self.label_map))]
        per_class_report = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        # ROC Curves
        auc_scores = plot_roc_curves(
            y_true=y_true,
            y_probs=y_probs,
            class_names=class_names,
            save_dir=self.config.roc_curves_dir,
        )

        # Misclassified Samples
        # Try to extract paths from dataset if present
        dataset = getattr(self.test_loader, "dataset", None)
        if dataset and hasattr(dataset, "df") and "processed_image_path" in dataset.df.columns:
            all_paths = dataset.df["processed_image_path"].tolist()
        else:
            all_paths = [f"Sample_{i}" for i in range(len(y_true))]

        misclassified = extract_misclassified_samples(y_true, y_pred, y_probs, all_paths, class_names)
        plot_misclassified_grid(misclassified, save_path=Path(self.config.misclassified_dir) / "grid.png")

        # Benchmark
        benchmark_results = benchmark_inference(self.model, device=self.device.type)

        results = {
            "global_metrics": global_metrics,
            "per_class_report": per_class_report,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "auc_scores": auc_scores,
            "class_names": class_names,
            "misclassified_count": len(misclassified),
            "benchmark": benchmark_results,
            "total_test_samples": len(y_true),
        }

        return results

    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], save_path: Path) -> None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0], cbar=False)
        axes[0].set_title("Brain MRI Confusion Matrix (Counts)")
        axes[0].set_xlabel("Predicted Disease")
        axes[0].set_ylabel("True Disease")

        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", xticklabels=class_names, yticklabels=class_names, ax=axes[1], cbar=False)
        axes[1].set_title("Brain MRI Confusion Matrix (Normalized)")
        axes[1].set_xlabel("Predicted Disease")
        axes[1].set_ylabel("True Disease")

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix heatmap to {save_path}")

    def save_reports(self, results: Dict[str, Any]) -> Tuple[Path, Path]:
        json_path = Path(self.config.classification_report_json)
        md_path = Path(self.config.performance_report_md)
        cm_png_path = Path(self.config.confusion_matrix_png)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        cm = np.array(results["confusion_matrix"])
        self.plot_confusion_matrix(cm, results["class_names"], cm_png_path)

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Build Markdown
        gm = results["global_metrics"]
        bm = results["benchmark"]
        md_lines = [
            "# Module 7: Brain MRI Disease Analysis Performance Report",
            "",
            "## 1. Global Classification Performance",
            "",
            f"- **Total Test Samples:** {results['total_test_samples']:,}",
            f"- **Accuracy:** {gm['accuracy']*100:.2f}%",
            f"- **Macro Precision:** {gm['precision_macro']:.4f}",
            f"- **Macro Recall:** {gm['recall_macro']:.4f}",
            f"- **Macro F1:** {gm['f1_macro']:.4f}",
            f"- **Weighted F1:** {gm['f1_weighted']:.4f}",
            f"- **Top-3 Accuracy:** {gm.get('top_3_accuracy', 1.0)*100:.2f}%",
            "",
            "## 2. Per-Disease Performance & AUC Scores",
            "",
            "| Disease Class | Precision | Recall | F1-Score | Support | ROC AUC |",
            "|---|---|---|---|---|---|",
        ]

        per_cls = results["per_class_report"]
        aucs = results["auc_scores"]
        for cls_name in results["class_names"]:
            if cls_name in per_cls:
                c = per_cls[cls_name]
                auc_val = aucs.get(cls_name, 0.0)
                md_lines.append(
                    f"| `{cls_name}` | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {int(c['support']):,} | {auc_val:.4f} |"
                )

        md_lines.extend([
            "",
            "## 3. Inference Benchmark & Latency",
            "",
            f"- **Device:** `{bm['device']}`",
            f"- **Mean Latency:** {bm['mean_latency_ms']} ms",
            f"- **p50 Latency:** {bm['p50_latency_ms']} ms",
            f"- **p95 Latency:** {bm['p95_latency_ms']} ms",
            f"- **Throughput:** {bm['throughput_fps']} FPS",
            f"- **Misclassified Samples:** {results['misclassified_count']}",
            "",
            "---",
            "*Report generated automatically by Module 7 Evaluation Engine.*",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved evaluation reports to {json_path} and {md_path}")
        return json_path, md_path
