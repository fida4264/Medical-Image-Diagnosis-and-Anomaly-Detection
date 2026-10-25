"""
ood_detection/msp.py
====================
Maximum Softmax Probability (MSP) complement scoring for OOD detection.
Reference: Hendrycks & Gimpel, "A Baseline for Detecting Misclassified and
Out-of-Distribution Examples in Neural Networks", ICLR 2017.

Orientation: HIGHER score = MORE OOD / ANOMALOUS.
  Score_MSP = 1.0 - max_c P(Y=c|X)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_msp_score(
    logits: torch.Tensor | np.ndarray,
    task: str = "multiclass",
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Computes MSP complement OOD score.

    Args:
        logits: Logit tensor or array of shape (N, C).
        task: 'multiclass', 'binary', or 'multilabel'.
        temperature: Optional temperature scaling parameter (default 1.0).

    Returns:
        1D numpy array of shape (N,) containing OOD scores in [0, 1].
        Higher score indicates higher OOD likelihood.
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.float()

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled_logits = logits_t / temperature

    if task == "multilabel":
        # Multi-label Retina: independent sigmoid heads (NOT softmax)
        probs = torch.sigmoid(scaled_logits)
        # Maximum positive disease/normal activation
        max_prob, _ = probs.max(dim=-1)
        # Score is complement: low maximum activation indicates OOD input
        scores = 1.0 - max_prob
    else:
        # Multiclass or binary: softmax over classes
        probs = F.softmax(scaled_logits, dim=-1)
        max_prob, _ = probs.max(dim=-1)
        scores = 1.0 - max_prob

    scores_np = scores.detach().cpu().numpy().astype(np.float64)
    # Ensure numerical bounds in [0, 1]
    return np.clip(scores_np, 0.0, 1.0)
