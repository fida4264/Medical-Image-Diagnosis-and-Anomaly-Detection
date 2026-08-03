"""
Medical AI Engine — Grad-CAM Target Selector

Module 15.3: Grad-CAM Engine

Resolves target classification category index and constructs target scalar/tensor
for backpropagation.
"""

from __future__ import annotations

import logging
from typing import Any, Tuple
import numpy as np

logger = logging.getLogger("TargetSelector")


class TargetSelector:
    """
    Selects and prepares target class score for PyTorch backward pass.
    """

    def resolve_target(
        self,
        output: Any,
        target_category: int | None = None,
    ) -> Tuple[int, Any]:
        """
        Determine target category index and extract output score tensor.

        Args:
            output: Model forward output tensor (e.g. logits or probabilities).
            target_category: Explicit requested class index. If None, auto-select argmax.

        Returns:
            Tuple of (category_index: int, category_score_scalar_or_tensor).
        """
        # PyTorch Tensor check
        if hasattr(output, "ndim") and hasattr(output, "argmax"):
            if output.ndim == 2:
                # [BatchSize, NumClasses]
                if target_category is None:
                    cat_idx = int(output.argmax(dim=-1)[0].item())
                else:
                    cat_idx = target_category
                score = output[0, cat_idx]
            elif output.ndim == 1:
                if target_category is None:
                    cat_idx = int(output.argmax(dim=-1).item())
                else:
                    cat_idx = target_category
                score = output[cat_idx]
            else:
                cat_idx = target_category or 0
                score = output

            logger.debug("[TargetSelector] Resolved target category: %d", cat_idx)
            return cat_idx, score

        # Numpy Array fallback
        if isinstance(output, np.ndarray):
            if output.ndim == 2:
                cat_idx = target_category if target_category is not None else int(np.argmax(output[0]))
                score = output[0, cat_idx]
            elif output.ndim == 1:
                cat_idx = target_category if target_category is not None else int(np.argmax(output))
                score = output[cat_idx]
            else:
                cat_idx = target_category or 0
                score = float(np.sum(output))
            return cat_idx, score

        # Fallback for scalar/mock
        cat_idx = target_category or 0
        return cat_idx, output
