"""
calibration/v2_phase7/temperature_scaling.py
=============================================
Post-hoc temperature scaling for confidence calibration.

Implements single-parameter temperature scaling (Guo et al., 2017):
  calibrated_prob = softmax(logits / T)

Temperature T is optimized by minimizing Negative Log-Likelihood (NLL)
on the VALIDATION set only. The test set is never used for fitting.

Outputs per modality (before and after calibration):
  - Negative Log-Likelihood (NLL)
  - Expected Calibration Error (ECE, 15-bin)
  - Maximum Calibration Error (MCE, 15-bin)
  - Brier Score
  - Reliability diagram data (bin_confidence, bin_accuracy, bin_count)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import LBFGS

logger = logging.getLogger(__name__)

N_BINS = 15  # Standard for ECE (Guo et al., 2017)


class TemperatureScaler(nn.Module):
    """
    Single-parameter temperature scaling for post-hoc calibration.
    Temperature T is a single scalar that divides the logits before softmax.
    Fit only on validation logits. Never touches the test set.
    """

    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """Returns logits divided by temperature (scaled logits)."""
        return logits / self.temperature.clamp(min=1e-4)

    def fit(
        self,
        val_logits: torch.Tensor,
        val_labels: torch.Tensor,
        max_iter: int = 200,
    ) -> float:
        """
        Fits temperature T by minimizing NLL on validation logits.

        Args:
            val_logits: Raw model logits from validation set, shape (N, C)
            val_labels: True class indices, shape (N,)
            max_iter: Maximum LBFGS iterations

        Returns:
            Optimal temperature value (float)
        """
        self.train()
        optimizer = LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        criterion = nn.CrossEntropyLoss()

        val_logits = val_logits.detach()
        val_labels = val_labels.detach()

        def closure():
            optimizer.zero_grad()
            scaled = self.forward(val_logits)
            loss = criterion(scaled, val_labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        self.eval()

        T = self.temperature.item()
        logger.info(f"Temperature fitted: T = {T:.6f}")
        return T

    def calibrate_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Returns calibrated softmax probabilities.

        Args:
            logits: Raw model logits, shape (N, C)

        Returns:
            Calibrated probability tensor, shape (N, C)
        """
        with torch.no_grad():
            scaled = self.forward(logits)
            return F.softmax(scaled, dim=-1)

    def save(self, path: Path | str) -> None:
        """Saves temperature parameter and metadata to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature.item(),
            "method": "temperature_scaling",
            "reference": "Guo et al. 2017 - On Calibration of Modern Neural Networks",
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Temperature parameters saved to {path}")

    @classmethod
    def load(cls, path: Path | str) -> "TemperatureScaler":
        """Loads temperature from a saved JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        scaler = cls()
        T = data["temperature"]
        scaler.temperature = nn.Parameter(torch.tensor([T]))
        scaler.eval()
        return scaler


# ---------------------------------------------------------------------------
# Calibration metric functions
# ---------------------------------------------------------------------------

def compute_nll(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Negative Log-Likelihood for multiclass predictions.
    NLL = -mean_n [ log(p_{n, y_n}) ]

    Args:
        probs: Probability array, shape (N, C)
        labels: Integer label array, shape (N,)

    Returns:
        Mean NLL (float)
    """
    eps = 1e-10
    n = len(labels)
    true_probs = probs[np.arange(n), labels.astype(int)]
    return float(-np.mean(np.log(np.clip(true_probs, eps, 1.0))))


def compute_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    """
    Multiclass Brier score: mean squared error between probability vector and one-hot label.
    Brier = mean_n [ sum_k (p_{n,k} - y_{n,k})^2 ]

    Args:
        probs: Probability array, shape (N, C)
        labels: Integer label array, shape (N,)
        num_classes: Number of classes C

    Returns:
        Mean Brier score (float)
    """
    n = len(labels)
    one_hot = np.zeros((n, num_classes), dtype=np.float32)
    one_hot[np.arange(n), labels.astype(int)] = 1.0
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def compute_ece_mce(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = N_BINS,
) -> Tuple[float, float, List[Dict[str, Any]]]:
    """
    Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    Uses equal-width confidence bins.

    Args:
        probs: Probability array, shape (N, C)
        labels: Integer label array, shape (N,)
        n_bins: Number of confidence bins

    Returns:
        (ECE, MCE, reliability_data)
        reliability_data: list of dicts with bin_lower, bin_upper,
                          bin_confidence, bin_accuracy, bin_count
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    correct = (predictions == labels.astype(int)).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(labels)
    reliability = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        # Edge case: include 0.0 in first bin
        if i == 0:
            in_bin = (confidences >= lo) & (confidences <= hi)

        count = int(in_bin.sum())
        if count == 0:
            reliability.append({
                "bin_lower": round(float(lo), 4),
                "bin_upper": round(float(hi), 4),
                "bin_confidence": None,
                "bin_accuracy": None,
                "bin_count": 0,
            })
            continue

        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(correct[in_bin].mean())
        gap = abs(avg_conf - avg_acc)

        ece += (count / n) * gap
        mce = max(mce, gap)

        reliability.append({
            "bin_lower": round(float(lo), 4),
            "bin_upper": round(float(hi), 4),
            "bin_confidence": round(avg_conf, 6),
            "bin_accuracy": round(avg_acc, 6),
            "bin_count": count,
        })

    return float(ece), float(mce), reliability


def run_calibration_for_modality(
    modality: str,
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    test_logits: Optional[torch.Tensor],
    test_labels: Optional[torch.Tensor],
    num_classes: int,
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Fits temperature scaling on validation logits and computes
    before/after calibration metrics for a single modality.

    IMPORTANT: test_logits/test_labels are used ONLY for final evaluation
    after the temperature has been frozen. They are NOT used to fit T.

    Args:
        modality: Modality name (for logging/saving)
        val_logits: Validation logits (N_val, C)
        val_labels: Validation labels (N_val,)
        test_logits: Optional test logits (N_test, C) — evaluated after fitting
        test_labels: Optional test labels (N_test,) — evaluated after fitting
        num_classes: Number of classes
        output_dir: Directory to save reliability data and temperature

    Returns:
        Dictionary with all before/after calibration metrics
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- BASELINE METRICS (uncalibrated) ---
    val_probs_base = F.softmax(val_logits, dim=-1).numpy()
    val_labels_np = val_labels.numpy()

    base_val_nll = compute_nll(val_probs_base, val_labels_np)
    base_val_brier = compute_brier_score(val_probs_base, val_labels_np, num_classes)
    base_val_ece, base_val_mce, base_val_reliability = compute_ece_mce(val_probs_base, val_labels_np)

    # --- FIT TEMPERATURE ON VALIDATION ONLY ---
    scaler = TemperatureScaler()
    temperature = scaler.fit(val_logits, val_labels)

    # --- CALIBRATED METRICS (validation) ---
    val_probs_cal = scaler.calibrate_probs(val_logits).numpy()
    cal_val_nll = compute_nll(val_probs_cal, val_labels_np)
    cal_val_brier = compute_brier_score(val_probs_cal, val_labels_np, num_classes)
    cal_val_ece, cal_val_mce, cal_val_reliability = compute_ece_mce(val_probs_cal, val_labels_np)

    # --- SAVE TEMPERATURE ---
    scaler.save(output_dir / f"{modality}_temperature.json")

    # --- SAVE RELIABILITY DATA ---
    import csv
    rel_path = output_dir / f"{modality}_reliability.csv"
    with open(rel_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "modality", "calibrated", "bin_lower", "bin_upper",
            "bin_confidence", "bin_accuracy", "bin_count",
        ])
        writer.writeheader()
        for row in base_val_reliability:
            writer.writerow({"modality": modality, "calibrated": False, **row})
        for row in cal_val_reliability:
            writer.writerow({"modality": modality, "calibrated": True, **row})

    result = {
        "modality": modality,
        "temperature": round(temperature, 6),
        "val_nll_before": round(base_val_nll, 6),
        "val_nll_after": round(cal_val_nll, 6),
        "val_ece_before": round(base_val_ece, 6),
        "val_ece_after": round(cal_val_ece, 6),
        "val_mce_before": round(base_val_mce, 6),
        "val_mce_after": round(cal_val_mce, 6),
        "val_brier_before": round(base_val_brier, 6),
        "val_brier_after": round(cal_val_brier, 6),
    }

    # --- FINAL TEST EVALUATION (after temperature is frozen) ---
    if test_logits is not None and test_labels is not None:
        test_labels_np = test_labels.numpy()
        test_probs_base = F.softmax(test_logits, dim=-1).numpy()
        test_probs_cal = scaler.calibrate_probs(test_logits).numpy()

        result.update({
            "test_nll_before": round(compute_nll(test_probs_base, test_labels_np), 6),
            "test_nll_after": round(compute_nll(test_probs_cal, test_labels_np), 6),
            "test_ece_before": round(compute_ece_mce(test_probs_base, test_labels_np)[0], 6),
            "test_ece_after": round(compute_ece_mce(test_probs_cal, test_labels_np)[0], 6),
            "test_mce_before": round(compute_ece_mce(test_probs_base, test_labels_np)[1], 6),
            "test_mce_after": round(compute_ece_mce(test_probs_cal, test_labels_np)[1], 6),
            "test_brier_before": round(compute_brier_score(test_probs_base, test_labels_np, num_classes), 6),
            "test_brier_after": round(compute_brier_score(test_probs_cal, test_labels_np, num_classes), 6),
        })

    logger.info(
        f"[{modality}] T={temperature:.4f} | "
        f"Val ECE: {base_val_ece:.4f} → {cal_val_ece:.4f} | "
        f"Val NLL: {base_val_nll:.4f} → {cal_val_nll:.4f}"
    )
    return result
