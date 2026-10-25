"""
evaluation/v2_phase6/error_analyzer.py
======================================
Comprehensive error characterization, uncalibrated confidence analysis,
minority-class failure isolation, and multi-label breakdown.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from training.v2_baseline.config import MODALITY_METADATA, ROOT

logger = logging.getLogger("error_analyzer")


class ErrorAnalyzer:
    """Performs deep error analysis on validated sample predictions."""

    def __init__(self, sample_predictions: List[Dict[str, Any]], output_dir: Path):
        self.samples = sample_predictions
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cm_dir = self.output_dir / "confusion_matrices"
        self.cm_dir.mkdir(parents=True, exist_ok=True)
        self.err_dir = self.output_dir / "error_analysis"
        self.err_dir.mkdir(parents=True, exist_ok=True)

    def extract_error_samples(self) -> pd.DataFrame:
        """Filters misclassified samples and writes error_samples.csv."""
        error_records = [s for s in self.samples if not s["correct"]]
        err_df = pd.DataFrame(error_records)
        out_csv = self.output_dir / "error_samples.csv"
        err_df.to_csv(out_csv, index=False)
        logger.info(f"Extracted {len(err_df)} error samples across all modalities to {out_csv}")
        return err_df

    def analyze_confidence_distributions(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyzes prediction scores (uncalibrated model probabilities).
        Compares distributions between correct and incorrect predictions.
        Identifies high-confidence error rates (overconfidence).
        """
        rows = []
        findings = {}

        df = pd.DataFrame(self.samples)

        # Exclude multilabel from scalar score comparisons where exact match is required
        # or analyze per-modality
        for modality in df["modality"].unique():
            mod_df = df[df["modality"] == modality]
            correct_scores = mod_df[mod_df["correct"]]["prediction_score"].values
            incorrect_scores = mod_df[~mod_df["correct"]]["prediction_score"].values

            def get_stats(scores):
                if len(scores) == 0:
                    return {
                        "count": 0, "mean": 0.0, "std": 0.0,
                        "median": 0.0, "min": 0.0, "max": 0.0,
                        "pct_above_80": 0.0, "pct_above_90": 0.0
                    }
                return {
                    "count": len(scores),
                    "mean": round(float(np.mean(scores)), 4),
                    "std": round(float(np.std(scores)), 4),
                    "median": round(float(np.median(scores)), 4),
                    "min": round(float(np.min(scores)), 4),
                    "max": round(float(np.max(scores)), 4),
                    "pct_above_80": round(float(np.mean(scores >= 0.80) * 100), 2),
                    "pct_above_90": round(float(np.mean(scores >= 0.90) * 100), 2),
                }

            c_stats = get_stats(correct_scores)
            inc_stats = get_stats(incorrect_scores)

            rows.append({
                "modality": modality,
                "subset": "correct",
                "sample_count": c_stats["count"],
                "mean_score": c_stats["mean"],
                "std_score": c_stats["std"],
                "median_score": c_stats["median"],
                "min_score": c_stats["min"],
                "max_score": c_stats["max"],
                "pct_above_80": c_stats["pct_above_80"],
                "pct_above_90": c_stats["pct_above_90"],
            })

            rows.append({
                "modality": modality,
                "subset": "incorrect",
                "sample_count": inc_stats["count"],
                "mean_score": inc_stats["mean"],
                "std_score": inc_stats["std"],
                "median_score": inc_stats["median"],
                "min_score": inc_stats["min"],
                "max_score": inc_stats["max"],
                "pct_above_80": inc_stats["pct_above_80"],
                "pct_above_90": inc_stats["pct_above_90"],
            })

            findings[modality] = {
                "correct": c_stats,
                "incorrect": inc_stats,
                "overconfidence_rate_80": inc_stats["pct_above_80"],
            }

        conf_df = pd.DataFrame(rows)
        out_csv = self.output_dir / "confidence_analysis.csv"
        conf_df.to_csv(out_csv, index=False)
        logger.info(f"Saved confidence distribution analysis to {out_csv}")
        return conf_df, findings

    def analyze_retina_multilabel(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Deep error analysis for Retina across all 8 labels.
        Evaluates support, TP, TN, FP, FN, precision, recall, F1, AUROC.
        Computes label co-occurrence.
        """
        retina_samples = [s for s in self.samples if s["modality"] == "retina" and s["split"] == "test"]
        if not retina_samples:
            retina_samples = [s for s in self.samples if s["modality"] == "retina"]

        class_names = MODALITY_METADATA["retina"]["class_names"]
        y_true = np.array([s["true_label"] for s in retina_samples], dtype=np.int32)
        y_pred = np.array([s["predicted_label"] for s in retina_samples], dtype=np.int32)
        y_prob = np.array([s["probabilities"] for s in retina_samples], dtype=np.float32)

        from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

        rows = []
        detailed_findings = {}

        for j, name in enumerate(class_names):
            t = y_true[:, j]
            p = y_pred[:, j]
            prob = y_prob[:, j]

            tp = int(np.sum((t == 1) & (p == 1)))
            tn = int(np.sum((t == 0) & (p == 0)))
            fp = int(np.sum((t == 0) & (p == 1)))
            fn = int(np.sum((t == 1) & (p == 0)))
            support = int(np.sum(t == 1))

            prec = precision_score(t, p, zero_division=0)
            rec = recall_score(t, p, zero_division=0)
            f1 = f1_score(t, p, zero_division=0)
            try:
                auc = roc_auc_score(t, prob) if len(np.unique(t)) > 1 else 0.5
            except Exception:
                auc = 0.5

            rows.append({
                "label": name,
                "support": support,
                "support_pct": round(support / len(y_true) * 100, 2),
                "true_positives": tp,
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(float(prec), 4),
                "recall": round(float(rec), 4),
                "f1_score": round(float(f1), 4),
                "auroc": round(float(auc), 4),
            })

            detailed_findings[name] = {
                "support": support, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
                "precision": round(float(prec), 4), "recall": round(float(rec), 4),
                "f1": round(float(f1), 4), "auroc": round(float(auc), 4)
            }

        retina_df = pd.DataFrame(rows)
        out_csv = self.output_dir / "retina_multilabel_analysis.csv"
        retina_df.to_csv(out_csv, index=False)

        # Label co-occurrence matrix
        co_occurrence = np.dot(y_true.T, y_true)
        co_df = pd.DataFrame(co_occurrence, index=class_names, columns=class_names)
        co_csv = self.err_dir / "retina_label_co_occurrence.csv"
        co_df.to_csv(co_csv)

        logger.info(f"Saved retina multi-label analysis to {out_csv} and {co_csv}")
        return retina_df, {"per_label": detailed_findings, "co_occurrence": co_df.to_dict()}

    def analyze_skin_imbalance(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Analyzes Skin Lesion class imbalance and determines whether minority
        classes drive the lower macro-F1 score.
        """
        skin_samples = [s for s in self.samples if s["modality"] == "skin" and s["split"] == "test"]
        if not skin_samples:
            skin_samples = [s for s in self.samples if s["modality"] == "skin"]

        class_names = MODALITY_METADATA["skin"]["class_names"]
        y_true = np.array([s["true_label_idx"] for s in skin_samples])
        y_pred = np.array([s["predicted_label_idx"] for s in skin_samples])

        from sklearn.metrics import classification_report, confusion_matrix

        rep = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))

        # Save confusion matrix CSV
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_df.to_csv(self.cm_dir / "skin_confusion_matrix.csv")

        rows = []
        minority_classes = ["dermatofibroma", "vascular_lesion", "actinic_keratosis", "basal_cell_carcinoma", "melanoma"]
        majority_classes = ["melanocytic_nevus", "benign_keratosis"]

        min_f1s = []
        maj_f1s = []

        for i, c in enumerate(class_names):
            m = rep[c]
            fp = int(np.sum((y_pred == i) & (y_true != i)))
            fn = int(np.sum((y_pred != i) & (y_true == i)))
            support = int(m["support"])
            f1 = float(m["f1-score"])

            is_minority = c in minority_classes
            if is_minority:
                min_f1s.append(f1)
            else:
                maj_f1s.append(f1)

            rows.append({
                "class_name": c,
                "is_minority": is_minority,
                "support": support,
                "support_pct": round(support / len(y_true) * 100, 2),
                "precision": round(float(m["precision"]), 4),
                "recall": round(float(m["recall"]), 4),
                "f1_score": round(f1, 4),
                "false_positives": fp,
                "false_negatives": fn,
            })

        skin_df = pd.DataFrame(rows)
        out_csv = self.output_dir / "skin_class_analysis.csv"
        skin_df.to_csv(out_csv, index=False)

        imbalance_summary = {
            "minority_macro_f1": round(float(np.mean(min_f1s)), 4),
            "majority_macro_f1": round(float(np.mean(maj_f1s)), 4),
            "macro_f1_gap": round(float(np.mean(maj_f1s) - np.mean(min_f1s)), 4),
            "minority_classes": minority_classes,
            "majority_classes": majority_classes,
        }

        logger.info(f"Saved skin lesion imbalance analysis to {out_csv}: {imbalance_summary}")
        return skin_df, imbalance_summary

    def generate_all_confusion_matrices(self) -> Dict[str, Dict[str, Any]]:
        """
        Extracts confusion matrices for all multiclass/binary modalities.
        Identifies most frequently confused pairs and asymmetry.
        """
        findings = {}
        for modality, meta in MODALITY_METADATA.items():
            if meta["task"] == "multilabel":
                continue

            mod_samples = [s for s in self.samples if s["modality"] == modality]
            # Use test split if available, else validation/fold
            test_samples = [s for s in mod_samples if s["split"] == "test"]
            eval_samples = test_samples if test_samples else mod_samples

            class_names = meta["class_names"]
            y_true = np.array([s["true_label_idx"] for s in eval_samples])
            y_pred = np.array([s["predicted_label_idx"] for s in eval_samples])

            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
            cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
            cm_csv = self.cm_dir / f"{modality}_confusion_matrix.csv"
            cm_df.to_csv(cm_csv)

            # Find top error pairs (off-diagonal)
            confused_pairs = []
            for i in range(len(class_names)):
                for j in range(len(class_names)):
                    if i != j and cm[i, j] > 0:
                        confused_pairs.append({
                            "true_class": class_names[i],
                            "pred_class": class_names[j],
                            "count": int(cm[i, j]),
                            "rate_of_true": round(float(cm[i, j] / max(1, np.sum(cm[i, :]))), 4)
                        })

            confused_pairs.sort(key=lambda x: x["count"], reverse=True)

            findings[modality] = {
                "matrix": cm.tolist(),
                "class_names": class_names,
                "top_confusions": confused_pairs[:5],
                "evaluated_samples": len(eval_samples),
            }

        logger.info(f"Generated confusion matrices for all modalities in {self.cm_dir}")
        return findings
