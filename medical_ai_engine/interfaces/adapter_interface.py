"""
Medical AI Engine — Adapter Factory & Loader Interfaces

Abstract contracts for Adapter Loader and Adapter Factory components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter


class IAdapterLoader(ABC):
    """
    Abstract interface for loading adapter instances and model checkpoints.
    """

    @abstractmethod
    def load(
        self,
        adapter_cls: Type[IPredictorAdapter],
        checkpoint_path: Optional[str] = None,
        label_map_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> IPredictorAdapter:
        """Instantiate and load model weights for an adapter class."""
        pass


class IAdapterFactory(ABC):
    """
    Abstract interface for creating adapter instances.
    """

    @abstractmethod
    def create_adapter(
        self,
        modality: str,
        adapter_cls: Type[IPredictorAdapter],
        config: Optional[Dict[str, Any]] = None,
    ) -> IPredictorAdapter:
        """Create an uninitialized or initialized adapter instance."""
        pass
