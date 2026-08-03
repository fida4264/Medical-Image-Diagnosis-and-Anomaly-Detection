"""
integration_validator.py — Module 15.6: Explainability Engine Integration
Medical AI Platform — Validates integration inputs and outputs for correctness.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from .integration_exceptions import IntegrationValidationError

logger = logging.getLogger(__name__)


class IntegrationValidator:
    """
    Validates ExplainabilityIntegrator inputs and outputs.

    Checks
    ------
    - EngineContext has the minimum required fields
    - ExplainabilityResult has a valid status
    - PredictionResult extension dict contains required keys
    - Modality is in supported set
    """

    REQUIRED_CONTEXT_FIELDS: List[str] = [
        "request_id",
        "session_id",
    ]

    VALID_EXPLAINABILITY_STATUSES = {"SUCCESS", "COMPLETED_WITH_WARNINGS", "FAILED", "SKIPPED"}

    def __init__(self, supported_modalities: Optional[List[str]] = None) -> None:
        self._supported_modalities = supported_modalities or [
            "brain_mri", "chest_xray", "brain_ct",
            "histopathology", "retina", "skin", "ultrasound", "unknown",
        ]

    # ── Engine Context Validation ─────────────────────────────────────────────

    def validate_engine_context(self, ctx: Any) -> None:
        """
        Validate that *ctx* is a valid EngineContext for explainability dispatch.

        Raises
        ------
        IntegrationValidationError
            If required fields are missing or state is invalid.
        """
        for fld in self.REQUIRED_CONTEXT_FIELDS:
            val = getattr(ctx, fld, None)
            if not val:
                raise IntegrationValidationError(
                    f"EngineContext missing required field: '{fld}'",
                    request_id=getattr(ctx, "request_id", ""),
                )

        state = getattr(ctx, "execution_state", "")
        if state not in {"COMPLETED", "RUNNING"}:
            raise IntegrationValidationError(
                f"EngineContext in unexpected state '{state}' for explainability dispatch.",
                request_id=getattr(ctx, "request_id", ""),
            )

    # ── ExplainabilityResult Validation ──────────────────────────────────────

    def validate_explainability_result(self, result: Any) -> None:
        """
        Validate ExplainabilityResult structure.

        Raises
        ------
        IntegrationValidationError
            If the result is None or has an invalid status.
        """
        if result is None:
            raise IntegrationValidationError("ExplainabilityResult is None.")

        status = getattr(result, "status", None)
        if status not in self.VALID_EXPLAINABILITY_STATUSES:
            raise IntegrationValidationError(
                f"ExplainabilityResult has invalid status: '{status}'",
                request_id=getattr(result, "request_id", ""),
            )

    # ── Extension Dict Validation ─────────────────────────────────────────────

    def validate_extension_dict(self, ext: Dict[str, Any], required_keys: List[str]) -> None:
        """
        Validate that *ext* contains all *required_keys*.

        Raises
        ------
        IntegrationValidationError
            If any required key is missing.
        """
        missing = [k for k in required_keys if k not in ext]
        if missing:
            raise IntegrationValidationError(
                f"ExplainabilityResult extension dict missing keys: {missing}"
            )

    # ── Confidence Gate ───────────────────────────────────────────────────────

    def should_run_explainability(
        self,
        confidence: float,
        modality: str,
        threshold: float,
        supported_modalities: Optional[List[str]] = None,
    ) -> bool:
        """
        Return True if the prediction qualifies for explainability generation.

        Conditions:
          1. confidence >= threshold
          2. modality is in supported_modalities (or supported list is empty)
        """
        mods = supported_modalities or self._supported_modalities
        if mods and modality not in mods:
            logger.debug(
                "IntegrationValidator: modality '%s' not in supported set — skip.", modality
            )
            return False
        if confidence < threshold:
            logger.debug(
                "IntegrationValidator: confidence %.4f < threshold %.4f — skip.",
                confidence, threshold,
            )
            return False
        return True


__all__ = ["IntegrationValidator"]
