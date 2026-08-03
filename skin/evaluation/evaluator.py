"""
skin/evaluation/evaluator.py
==============================
Evaluator for Module 12: Skin Lesion Cancer Classification Framework.
Computes test metrics, per-class metrics, confusion matrices, ROC curves, and benchmarks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modality_classifier.training.metrics import compute_classification_metrics  # Module 6 reuse
from skin.evaluation.benchmark import BenchmarkEngine
from skin.evaluation.confusion_matrix import plot_confusion_matrix
from skin.evaluation.misclassified import save_misclassified_samples
from skin.evaluation.roc import plot_multiclass_roc
from skin.utils.config import EvaluationConfig

logger = logging.getLogger(__name__)


class SkinLesionEvaluator:
    """Evaluates trained skin lesion models on test DataLoaders."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_map: Dict[str, int],
        config: Optional[EvaluationConfig] = None,
    ):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.config = config or EvaluationConfig()
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, any]:
        """Run evaluation on test loader and return comprehensive metric dict."""
        self.model.eval()
        all_targets = []
        all_probs = []

        for images, labels in self.test_loader:
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            all_targets.extend(labels.numpy())
            all_probs.extend(probs)

        y_true = np.array(all_targets)
        y_probs = np.array(all_probs)
        y_pred = np.argmax(y_probs, axis=1)

        # Global metrics using Module 6 function
        metrics = compute_classification_metrics(y_true, y_probs)

        class_names = [self.inv_label_map.get(i, f"class_{i}") for i in range(len(self.label_map))]

        # Save ROC curve plot
        if self.config.generate_roc and len(class_names) > 1:
            roc_path = self.output_dir / "roc_curves.png"
            plot_multiclass_roc(y_true, y_probs, class_names, save_path=roc_path)

        # Save Confusion Matrix plot
        if self.config.generate_confusion_matrix:
            cm_path = self.output_dir / "confusion_matrix.png"
            plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_path)

        # Save Misclassified Grid
        if self.config.generate_misclassified:
            mis_path = self.output_dir / "misclassified_samples.png"
            save_misclassified_samples(
                self.test_loader.dataset, y_true, y_pred, class_names, save_path=mis_path
            )

        # Latency Benchmark
        benchmark_results = {}
        if self.config.run_benchmark:
            bm = BenchmarkEngine(self.model, iterations=self.config.benchmark_iterations)
            benchmark_results = bm.run()

        results = {
            "global_metrics": metrics,
            "class_names": class_names,
            "num_samples": len(y_true),
            "benchmark": benchmark_results,
        }

        logger.info(
            f"Evaluation complete. Accuracy: {metrics['accuracy']*100:.2f}%, "
            f"Macro F1: {metrics['f1_macro']:.4f}"
        )
        return results

    def save_reports(self, eval_results: Dict[str, any]) -> None:
        """Save performance evaluation reports to disk."""
        report_path = self.output_dir / "performance_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(eval_results, f, indent=2, default=str)
        logger.info(f"Evaluation report saved to '{report_path}'.")
