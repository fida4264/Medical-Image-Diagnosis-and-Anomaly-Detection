"""
Medical AI Engine — Grad-CAM Core Computation

Module 15.3: Grad-CAM Engine

Performs the mathematical aggregation of activation feature maps weighted by
global-average-pooled backward gradients (Selvaraju et al., 2017).

Formula:
  alpha_k = (1 / Z) * sum_{i,j} (grad_{k, i, j})
  cam = sum_k (alpha_k * act_k)
  cam = ReLU(cam)  [if relu_on_output is True]
"""

from __future__ import annotations

import logging
import numpy as np

from medical_ai_engine.explainability.plugins.gradcam.gradcam_exceptions import (
    CAMComputationError,
)

logger = logging.getLogger("CAMComputation")


class CAMComputation:
    """
    Computes raw Grad-CAM activation map from activations and gradients.
    """

    def compute(
        self,
        activations: np.ndarray,
        gradients: np.ndarray,
        relu_on_output: bool = True,
    ) -> np.ndarray:
        """
        Compute Grad-CAM activation map.

        Args:
            activations: Feature maps array of shape (B, C, H, W) or (C, H, W).
            gradients: Gradient array of shape (B, C, H, W) or (C, H, W).
            relu_on_output: Apply ReLU activation to final map.

        Returns:
            2D numpy float32 array of shape (H, W).
        """
        try:
            # Ensure 4D (B, C, H, W)
            acts = activations
            grads = gradients

            if acts.ndim == 3:
                acts = np.expand_dims(acts, axis=0)
            if grads.ndim == 3:
                grads = np.expand_dims(grads, axis=0)

            if acts.ndim != 4 or grads.ndim != 4:
                raise CAMComputationError(
                    f"Expected 4D arrays for Grad-CAM. Got activations {acts.shape}, gradients {grads.shape}"
                )

            # Take single batch item (index 0)
            act_b = acts[0]    # Shape (C, H, W)
            grad_b = grads[0]  # Shape (C, H, W)

            # 1. Global Average Pooling of gradients across spatial dimensions (H, W)
            weights = np.mean(grad_b, axis=(1, 2))  # Shape (C,)

            # 2. Weighted combination of activation channels
            # Shape (C, H, W) * (C, 1, 1) -> (C, H, W) -> sum over C -> (H, W)
            cam = np.sum(weights[:, None, None] * act_b, axis=0)

            # 3. Apply ReLU if requested (keeps positive contributions to decision)
            if relu_on_output:
                cam = np.maximum(cam, 0)

            logger.debug("[CAMComputation] Raw Grad-CAM map generated shape: %s", cam.shape)
            return cam.astype(np.float32)
        except CAMComputationError:
            raise
        except Exception as exc:
            raise CAMComputationError(f"Unexpected error computing Grad-CAM map: {exc}") from exc
