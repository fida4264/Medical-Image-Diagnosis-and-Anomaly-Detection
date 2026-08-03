"""
retina/evaluation/evaluator.py
================================
RetinaEvaluator — runs test-set evaluation, saves JSON + Markdown reports.
Mirrors Module 7/8/9/10 evaluator pattern.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from modality_classifier.training.metrics import compute_classification_metrics  # Module 6 reuse

from retina.evaluation.confusion_matrix import plot_confusion_matrix
from retina.evaluation.roc import plot_roc_curves
from retina.evaluation.misclassified import save_misclassified_grid
from retina.evaluation.benchmark import run_latency_benchmark
from retina.utils.config import EvaluationConfig

logger = logging.getLogger(__name__)


class RetinaEvaluator:
    """Evaluates the trained retina model on the test set."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        label_map: Dict[str, int],
        config: EvaluationConfig,
    ):
        self.model = model
        self.test_loader = test_loader
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}
        self.num_classes = len(label_map)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @torch.no_grad()
    def evaluate(self) -> Dict[str, Any]:
        """Run full test-set evaluation and return metrics dict."""
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        all_images = []

        t0 = time.time()
        for images, labels in self.test_loader:
            images_dev = images.to(self.device)
            logits = self.model(images_dev)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())
            all_probs.extend(probs.cpu().tolist())
            all_images.extend(images.tolist())

        elapsed = time.time() - t0

        global_metrics = compute_classification_metrics(all_labels, np.array(all_probs))
        logger.info(
            f"Retina Test Evaluation — Acc={global_metrics['accuracy']:.4f}, "
            f"F1={global_metrics['f1_macro']:.4f}, Time={elapsed:.2f}s"
        )

        per_class = {}
        for idx, class_name in self.inv_label_map.items():
            class_preds = [p == idx for p in all_preds]
            class_labels = [l == idx for l in all_labels]
            tp = sum(p and l for p, l in zip(class_preds, class_labels))
            fp = sum(p and not l for p, l in zip(class_preds, class_labels))
            fn = sum(not p and l for p, l in zip(class_preds, class_labels))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[class_name] = {"precision": prec, "recall": rec, "f1": f1}

        return {
            "global_metrics": global_metrics,
            "per_class_metrics": per_class,
            "all_preds": all_preds,
            "all_labels": all_labels,
            "all_probs": all_probs,
            "inference_time_s": elapsed,
            "num_samples": len(all_labels),
        }

    def save_reports(self, results: Dict[str, Any]) -> Tuple[Path, Path]:
        """Save evaluation JSON and Markdown performance reports, plus plots."""
        from retina.reports.performance_report import PerformanceReportGenerator

        # Confusion matrix
        if self.config.generate_confusion_matrix:
            try:
                cm_path = self.output_dir / "confusion_matrix.png"
                plot_confusion_matrix(
                    results["all_labels"],
                    results["all_preds"],
                    class_names=list(self.inv_label_map.values()),
                    output_path=cm_path,
                )
            except Exception as e:
                logger.warning(f"Confusion matrix plot failed: {e}")

        # ROC curves
        if self.config.generate_roc:
            try:
                roc_path = self.output_dir / "roc_curves.png"
                plot_roc_curves(
                    results["all_labels"],
                    results["all_probs"],
                    class_names=list(self.inv_label_map.values()),
                    output_path=roc_path,
                )
            except Exception as e:
                logger.warning(f"ROC curves plot failed: {e}")

        # Latency benchmark
        if self.config.run_benchmark:
            try:
                run_latency_benchmark(
                    self.model,
                    device=self.device,
                    output_path=self.output_dir / "benchmark.json",
                    iterations=self.config.benchmark_iterations,
                )
            except Exception as e:
                logger.warning(f"Benchmark failed: {e}")

        # Performance report
        json_path, md_path = PerformanceReportGenerator.save(results, self.label_map, self.output_dir)
        return json_path, md_path
