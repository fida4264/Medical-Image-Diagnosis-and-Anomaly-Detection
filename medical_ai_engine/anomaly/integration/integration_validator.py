"""
integration_validator.py — Module 16.6: Anomaly Engine Integration
Medical AI Platform — Validates inputs and readiness for post-prediction anomaly integration.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional
from .integration_exceptions import IntegrationValidationError

logger = logging.getLogger(__name__)


class IntegrationValidator:
    """
    Validates prediction inputs, context readiness, and modality eligibility for anomaly detection.
    """

    def __init__(self, supported_modalities: Optional[List[str]] = None) -> None:
        self._modalities = supported_modalities or [
            "brain_mri", "chest_xray", "histopathology",
            "retina", "skin", "ultrasound", "brain_ct",
        ]

    def validate_inputs(self, engine_ctx: Any, prediction_result: Any) -> None:
        """
        Validate engine context and prediction result objects.

        Raises
        ------
        IntegrationValidationError
            If prediction_result is None or missing required fields.
        """
        if prediction_result is None:
            raise IntegrationValidationError("prediction_result cannot be None.")
        if engine_ctx is None:
            raise IntegrationValidationError("engine_ctx cannot be None.")

        req_id = getattr(prediction_result, "request_id", None)
        if not req_id:
            logger.warning("prediction_result missing request_id attribute.")

    def is_eligible(self, prediction_result: Any) -> bool:
        """Check if prediction_result is eligible for anomaly detection."""
        if prediction_result is None:
            return False
        modality = getattr(prediction_result, "modality", "unknown")
        if modality not in self._modalities:
            logger.debug("Modality '%s' not in supported set %s", modality, self._modalities)
            return False
        return True


__all__ = ["IntegrationValidator"]
