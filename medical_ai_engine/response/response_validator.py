"""
Medical AI Engine — Response Validator

Validates PredictionResult for required fields, schema compliance,
and internal consistency.
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING
from medical_ai_engine.interfaces.response_validator_interface import IResponseValidator
from medical_ai_engine.response.response_configuration import ResponseConfiguration
from medical_ai_engine.response.response_exceptions import ResponseValidationException
from medical_ai_engine.response.response_schema import REQUIRED_FIELDS

if TYPE_CHECKING:
    from medical_ai_engine.response.prediction_result import PredictionResult


class ResponseValidator(IResponseValidator):
    """
    Validates a PredictionResult for completeness and consistency.
    """

    def __init__(self, config: Optional[ResponseConfiguration] = None) -> None:
        self.config = config or ResponseConfiguration.default()

    def validate(self, result: "PredictionResult") -> None:
        """
        Perform full validation; raise ResponseValidationException on failures.
        """
        errors = self.get_validation_errors(result)
        if errors and self.config.strict_validation:
            raise ResponseValidationException(
                message="PredictionResult validation failed.",
                validation_errors=errors,
            )

    def get_validation_errors(self, result: "PredictionResult") -> List[str]:
        """Collect all validation errors without raising."""
        errors: List[str] = []

        # 1. Required field presence
        d = result.to_dict()
        for req_field in REQUIRED_FIELDS:
            val = d.get(req_field)
            if val is None or val == "":
                errors.append(f"Required field '{req_field}' is missing or empty.")

        # 2. Confidence range
        if not (0.0 <= result.confidence <= 1.0):
            errors.append(
                f"confidence={result.confidence} is out of valid range [0.0, 1.0]."
            )

        # 3. request_id / session_id non-empty
        if not result.request_id or not result.request_id.strip():
            errors.append("request_id must be a non-empty string.")
        if not result.session_id or not result.session_id.strip():
            errors.append("session_id must be a non-empty string.")

        # 4. Disease prediction present for success statuses
        from medical_ai_engine.response.response_schema import PredictionStatus
        success_statuses = {PredictionStatus.SUCCESS, PredictionStatus.COMPLETED_WITH_WARNINGS}
        if result.status in success_statuses:
            if result.predicted_disease in ("", "unknown", None):
                errors.append(
                    "predicted_disease must be set for SUCCESS/COMPLETED_WITH_WARNINGS status."
                )
            if result.modality in ("", "unknown", None):
                errors.append(
                    "modality must be set for SUCCESS/COMPLETED_WITH_WARNINGS status."
                )

        return errors
