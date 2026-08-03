"""
Medical AI Engine — Explainability Engine Interface

Module 15.1: Explainability Framework Architecture

Defines the core interface for the Explainability Engine orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

if TYPE_CHECKING:
    from medical_ai_engine.explainability.explainability_context import ExplainabilityContext
    from medical_ai_engine.explainability.explainability_result import ExplainabilityResult


class IExplainabilityEngine(ABC):
    """
    Abstract interface for the ExplainabilityEngine subsystem.
    """

    @abstractmethod
    def explain(self, context: ExplainabilityContext) -> ExplainabilityResult:
        """
        Execute the explainability pipeline for a completed prediction context.

        Args:
            context: Populated ExplainabilityContext.

        Returns:
            Immutable ExplainabilityResult object.
        """
        pass

    @abstractmethod
    def register_cam_method(self, method_key: str, method_cls: Any) -> None:
        """Register a new CAM method plugin."""
        pass

    @abstractmethod
    def get_supported_methods(self) -> Dict[str, str]:
        """Return a mapping of registered CAM method keys to method titles."""
        pass
