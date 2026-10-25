"""
ood_detection/energy.py
=======================
Free Energy scoring for OOD detection.
Reference: Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.

Orientation: HIGHER score = MORE OOD / ANOMALOUS.
  In-distribution inputs have high activation -> strongly negative energy.
  OOD inputs have low activation -> higher (less negative) energy.
"""

from __future__ import annotations

import numpy as np
import torch


def compute_energy_score(
    logits: torch.Tensor | np.ndarray,
    task: str = "multiclass",
    temperature: float = 1.0,
) -> np.ndarray:
    """
    Computes Energy-based OOD score.

    For multiclass models: E(x; T) = -T * log(sum_c exp(z_c / T)) = -T * logsumexp(z / T).
    For multi-label Retina: Free energy of independent binary heads:
      E_multi(x; T) = -T * sum_c log(1 + exp(z_c / T)) = -T * sum_c softplus(z_c / T).
    Does NOT assume multiclass softmax for Retina.

    Args:
        logits: Logit tensor or array of shape (N, C).
        task: 'multiclass', 'binary', or 'multilabel'.
        temperature: Temperature hyperparameter T (default 1.0).

    Returns:
        1D numpy array of shape (N,) containing Energy OOD scores.
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
        # Free energy of independent binary units: -T * sum_c softplus(z_c / T)
        # softplus(x) = log(1 + exp(x))
        softplus_vals = torch.nn.functional.softplus(scaled_logits)
        energy = -temperature * softplus_vals.sum(dim=-1)
    else:
        # Multiclass: -T * logsumexp(z / T)
        energy = -temperature * torch.logsumexp(scaled_logits, dim=-1)

    scores_np = energy.detach().cpu().numpy().astype(np.float64)
    return scores_np
