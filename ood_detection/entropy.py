"""
ood_detection/entropy.py
========================
Predictive Entropy scoring for OOD detection.
Reference: Lakshminarayanan et al., "Simple and Scalable Predictive Uncertainty
Estimation using Deep Ensembles", NeurIPS 2017.

Orientation: HIGHER score = MORE OOD / UNCERTAIN.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def compute_entropy_score(
    logits: torch.Tensor | np.ndarray,
    task: str = "multiclass",
    temperature: float = 1.0,
    normalize: bool = True,
) -> np.ndarray:
    """
    Computes predictive entropy OOD score.

    For multiclass models: Shannon entropy over softmax probabilities.
    For multi-label Retina: Mean binary entropy over independent sigmoid heads.
    Does NOT apply multiclass softmax to multi-label models.

    Args:
        logits: Logit tensor or array of shape (N, C).
        task: 'multiclass', 'binary', or 'multilabel'.
        temperature: Optional temperature scaling parameter (default 1.0).
        normalize: Whether to normalize score to [0, 1].

    Returns:
        1D numpy array of shape (N,) containing entropy scores.
        Higher score indicates higher uncertainty / OOD likelihood.
    """
    if isinstance(logits, np.ndarray):
        logits_t = torch.from_numpy(logits).float()
    else:
        logits_t = logits.float()

    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled_logits = logits_t / temperature
    eps = 1e-12

    if task == "multilabel":
        # Multi-label Retina: independent sigmoid heads
        probs = torch.sigmoid(scaled_logits)
        probs = torch.clamp(probs, eps, 1.0 - eps)
        # Binary entropy per label: -[p*log(p) + (1-p)*log(1-p)]
        bin_entropy = -(probs * torch.log(probs) + (1.0 - probs) * torch.log(1.0 - probs))
        # Mean binary entropy across all labels
        mean_entropy = bin_entropy.mean(dim=-1)
        if normalize:
            # Binary entropy max is log(2)
            max_val = np.log(2.0)
            scores = mean_entropy / max_val
        else:
            scores = mean_entropy
    else:
        # Multiclass or binary: Shannon entropy over softmax
        probs = F.softmax(scaled_logits, dim=-1)
        probs = torch.clamp(probs, min=eps)
        entropy = -torch.sum(probs * torch.log(probs), dim=-1)
        num_classes = logits_t.shape[-1]
        if normalize and num_classes > 1:
            max_val = np.log(float(num_classes))
            scores = entropy / max_val
        else:
            scores = entropy

    scores_np = scores.detach().cpu().numpy().astype(np.float64)
    if normalize:
        scores_np = np.clip(scores_np, 0.0, 1.0)
    return scores_np


# Alias for explicit naming
compute_predictive_entropy_score = compute_entropy_score
