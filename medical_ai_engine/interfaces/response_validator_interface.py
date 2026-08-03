"""
Medical AI Engine — Response Validator Interface

Abstract contract for ResponseValidator implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.response.prediction_result import PredictionResult


class IResponseValidator(ABC):
    """
    Abstract interface for validating PredictionResult objects.
    """

    @abstractmethod
    def validate(self, result: "PredictionResult") -> None:
        """
        Validate the PredictionResult for required fields, schema compliance,
        and internal consistency.

        Args:
            result: PredictionResult to validate.

        Raises:
            ResponseValidationException: If validation fails.
        """
        pass

    @abstractmethod
    def get_validation_errors(self, result: "PredictionResult") -> List[str]:
        """
        Collect all validation errors without raising.

        Returns:
            List of validation error messages (empty if valid).
        """
        pass
