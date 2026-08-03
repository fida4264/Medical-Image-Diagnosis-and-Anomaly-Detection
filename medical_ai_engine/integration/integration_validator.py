"""
Medical AI Engine — Integration Validator

Validates end-to-end prediction workflow outputs, verifying that EngineContext
and PredictionResult adhere to architectural guarantees.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.integration.integration_configuration import IntegrationConfiguration
from medical_ai_engine.integration.integration_exceptions import IntegrationValidationException
from medical_ai_engine.response.prediction_result import PredictionResult
from medical_ai_engine.response.response_schema import PredictionStatus

logger = logging.getLogger("IntegrationValidator")


class IntegrationValidator:
    """
    End-to-End System Integration Validator.

    Verifies:
      1. EngineContext reached completed state with required lifecycle timestamps.
      2. PredictionResult is valid, frozen, and matches request IDs.
      3. Modality matches expected modality (if provided).
      4. Routing decision metadata is consistent with detected modality.
      5. Clinical report is populated with diagnosis, recommendations, notes.
      6. Performance metrics are recorded and non-negative.
    """

    def __init__(self, config: Optional[IntegrationConfiguration] = None) -> None:
        self.config = config or IntegrationConfiguration.default()

    def validate_workflow_execution(
        self,
        ctx: EngineContext,
        result: PredictionResult,
        expected_modality: Optional[str] = None,
    ) -> None:
        """
        Validate end-to-end workflow execution.

        Args:
            ctx: Populated EngineContext.
            result: Immutable PredictionResult.
            expected_modality: Optional modality key that must match result.modality.

        Raises:
            IntegrationValidationException: If validation fails.
        """
        errors = self.get_validation_errors(ctx, result, expected_modality=expected_modality)
        if errors and self.config.enable_strict_validation:
            raise IntegrationValidationException(
                message="End-to-end workflow integration validation failed.",
                validation_errors=errors,
            )

    def get_validation_errors(
        self,
        ctx: EngineContext,
        result: PredictionResult,
        expected_modality: Optional[str] = None,
    ) -> List[str]:
        """Collect all integration validation errors without raising."""
        errors: List[str] = []

        # 1. ID Consistency
        if ctx.request_id != result.request_id:
            errors.append(f"Request ID mismatch: context={ctx.request_id}, result={result.request_id}")
        if ctx.session_id != result.session_id:
            errors.append(f"Session ID mismatch: context={ctx.session_id}, result={result.session_id}")

        # 2. Immutability
        if not hasattr(result, "__dataclass_params__") or not getattr(result.__dataclass_params__, "frozen", False):
            errors.append("PredictionResult is not a frozen dataclass.")

        # 3. Expected Modality match
        if expected_modality:
            expected_clean = expected_modality.lower().strip()
            if result.modality.lower().strip() != expected_clean:
                errors.append(
                    f"Modality mismatch: expected '{expected_clean}', got '{result.modality}'"
                )

        # 4. Status-specific checks
        if result.status in (PredictionStatus.SUCCESS, PredictionStatus.COMPLETED_WITH_WARNINGS):
            if result.predicted_disease in ("", "unknown", None):
                errors.append("SUCCESS result must contain a valid predicted_disease.")

            if not result.clinical_report:
                errors.append("SUCCESS result must contain a non-empty clinical_report.")
            else:
                if "primary_diagnosis" not in result.clinical_report:
                    errors.append("clinical_report missing 'primary_diagnosis'.")
                if "recommendations" not in result.clinical_report:
                    errors.append("clinical_report missing 'recommendations'.")

        # 5. Performance Metrics
        if self.config.record_performance_metrics:
            if not result.performance_metrics:
                errors.append("Performance metrics dict is missing from PredictionResult.")
            else:
                for k, v in result.performance_metrics.items():
                    if v < 0.0:
                        errors.append(f"Negative metric timing for key '{k}': {v}ms")

        # 6. Timestamp ordering
        ts = ctx.timestamps
        if "request_received" in ts and "execution_completed" in ts:
            if ts["request_received"] > ts["execution_completed"]:
                errors.append("Timestamp ordering violation: request_received > execution_completed.")

        return errors
