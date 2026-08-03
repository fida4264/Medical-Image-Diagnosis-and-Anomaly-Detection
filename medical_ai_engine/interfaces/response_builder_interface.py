"""
Medical AI Engine — Response Builder Interface

Abstract contract for ResponseBuilder implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext
    from medical_ai_engine.response.prediction_result import PredictionResult


class IResponseBuilder(ABC):
    """
    Abstract interface for constructing immutable PredictionResult
    from a completed EngineContext.
    """

    @abstractmethod
    def build(self, ctx: "EngineContext") -> "PredictionResult":
        """
        Construct an immutable PredictionResult from a completed EngineContext.

        Args:
            ctx: Fully populated EngineContext after pipeline execution.

        Returns:
            Immutable PredictionResult instance.

        Raises:
            ResponseBuildException: If required fields are missing or invalid.
        """
        pass

    @abstractmethod
    def build_error_response(
        self,
        ctx: "EngineContext",
        error_message: str,
    ) -> "PredictionResult":
        """
        Construct a failure PredictionResult when the pipeline execution failed.

        Args:
            ctx: Partially populated EngineContext.
            error_message: Description of the failure reason.

        Returns:
            Immutable PredictionResult with FAILED status.
        """
        pass
