"""
modality_classifier/evaluation/evaluator.py
============================================
Evaluates Modality Classifier model on test set, generates confusion matrices, and writes reports.
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
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from modality_classifier.config import EvaluationConfig
from modality_classifier.training.metrics import compute_classification_metrics

logger = logging.getLogger(__name__)


class ModalityEvaluator:
    """
    Evaluator performing full evaluation on test dataset and generating charts + reports.
    """

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_map: Dict[str, int],
        config: Optional[EvaluationConfig] = None,
        device: Optional[str] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.inv_label_map = {idx: mod for mod, idx in label_map.items()}
        self.config = config or EvaluationConfig()

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """
        Runs evaluation on test set and returns comprehensive metrics dict.
        """
        all_probs = []
        all_targets = []

        for images, targets in self.test_loader:
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

        # Per-class classification report
        class_names = [self.inv_label_map.get(i, f"Class_{i}") for i in range(len(self.label_map))]
        report_dict = classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        )

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        results = {
            "global_metrics": global_metrics,
            "per_class_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "confusion_matrix_normalized": cm_norm.tolist(),
            "class_names": class_names,
            "total_test_samples": len(y_true),
        }

        return results

    def plot_confusion_matrix(
        self,
        cm: np.ndarray,
        class_names: List[str],
        save_path: str | Path,
        title: str = "Modality Classification Confusion Matrix",
    ) -> None:
        """
        Plots heatmaps for raw & normalized confusion matrices.
        """
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        cm_norm = cm.astype("float") / (cm.sum(axis=1, keepdims=True) + 1e-8)

        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Raw counts
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[0],
            cbar=False,
        )
        axes[0].set_title(f"{title} (Counts)")
        axes[0].set_xlabel("Predicted Modality")
        axes[0].set_ylabel("True Modality")
        axes[0].tick_params(axis="x", rotation=45)

        # Normalized
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt=".2f",
            cmap="Greens",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=axes[1],
            cbar=False,
        )
        axes[1].set_title(f"{title} (Normalized)")
        axes[1].set_xlabel("Predicted Modality")
        axes[1].set_ylabel("True Modality")
        axes[1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved confusion matrix plot to {path}")

    def save_reports(self, results: Dict[str, Any]) -> Tuple[Path, Path]:
        """
        Saves JSON and Markdown evaluation reports.
        """
        json_path = Path(self.config.report_json_path)
        md_path = Path(self.config.report_md_path)
        png_path = Path(self.config.confusion_matrix_png)

        json_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)

        # Plot confusion matrix
        cm = np.array(results["confusion_matrix"])
        class_names = results["class_names"]
        self.plot_confusion_matrix(cm, class_names, png_path)

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        # Build Markdown
        gm = results["global_metrics"]
        md_lines = [
            "# Module 6: Medical Image Modality Classification Report",
            "",
            "## 1. Global Performance Metrics",
            "",
            f"- **Test Set Size:** {results['total_test_samples']:,} images",
            f"- **Overall Accuracy:** {gm['accuracy']*100:.2f}%",
            f"- **Macro Precision:** {gm['precision_macro']:.4f}",
            f"- **Macro Recall:** {gm['recall_macro']:.4f}",
            f"- **Macro F1 Score:** {gm['f1_macro']:.4f}",
        ]

        if "top_3_accuracy" in gm:
            md_lines.append(f"- **Top-3 Accuracy:** {gm['top_3_accuracy']*100:.2f}%")

        md_lines.extend([
            "",
            "## 2. Per-Modality Breakdown",
            "",
            "| Modality | Precision | Recall | F1-Score | Support |",
            "|---|---|---|---|---|",
        ])

        per_class = results["per_class_report"]
        for cls_name in class_names:
            if cls_name in per_class:
                c = per_class[cls_name]
                md_lines.append(
                    f"| `{cls_name}` | {c['precision']:.4f} | {c['recall']:.4f} | {c['f1-score']:.4f} | {int(c['support']):,} |"
                )

        md_lines.extend([
            "",
            "## 3. Visualization Artifacts",
            "",
            f"![Confusion Matrix](file:///{png_path.as_posix()})",
            "",
            "---",
            "*Report generated automatically by Module 6 Evaluation Engine.*",
        ])

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))

        logger.info(f"Saved evaluation reports to {json_path} and {md_path}")
        return json_path, md_path
