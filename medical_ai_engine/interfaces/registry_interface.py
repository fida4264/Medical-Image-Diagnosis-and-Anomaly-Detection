"""
Medical AI Engine — Model Registry Interface

Abstract interface for the Model Registry component.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Type

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter


class IModelRegistry(ABC):
    """
    Abstract contract for ModelRegistry implementations.
    Declares registration, lookup, lazy loading, cache management, and health check APIs.
    """

    @abstractmethod
    def register_adapter(
        self,
        modality: str,
        adapter_cls: Type[IPredictorAdapter],
        metadata: Optional[Dict[str, Any]] = None,
        overwrite: bool = False,
    ) -> None:
        """Register a new predictor adapter class for a modality key."""
        pass

    @abstractmethod
    def get_adapter(self, modality: str) -> IPredictorAdapter:
        """
        Retrieve a loaded, ready-to-predict adapter instance for the modality.
        Triggers lazy loading on first access.
        """
        pass

    @abstractmethod
    def is_registered(self, modality: str) -> bool:
        """Returns True if an adapter is registered for the modality."""
        pass

    @abstractmethod
    def is_loaded(self, modality: str) -> bool:
        """Returns True if the adapter for the modality is currently loaded in memory."""
        pass

    @abstractmethod
    def list_registered_modalities(self) -> List[str]:
        """Return list of all registered modality keys."""
        pass

    @abstractmethod
    def invalidate_cache(self, modality: Optional[str] = None) -> None:
        """Evict cached adapter instance(s) from memory."""
        pass

    @abstractmethod
    def reload_adapter(self, modality: str) -> IPredictorAdapter:
        """Unload and reload model weights for a modality."""
        pass

    @abstractmethod
    def hot_swap(self, modality: str, new_checkpoint_path: str) -> IPredictorAdapter:
        """Zero-downtime hot-swap of model weights for a modality."""
        pass

    @abstractmethod
    def health_check(self) -> Dict[str, Any]:
        """Return comprehensive health assessment of all registered adapters."""
        pass
