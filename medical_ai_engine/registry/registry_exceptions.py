"""
Medical AI Engine — Registry Exception Hierarchy

Defines exception types for registry operations.
Inherits from EngineException (ENG-700 block).
"""

from __future__ import annotations

from typing import Optional
from medical_ai_engine.core.exceptions import RegistryException


class DuplicateAdapterRegistrationException(RegistryException):
    """Raised when registering an adapter for an already registered modality."""

    ERROR_CODE = "REG-101"

    def __init__(self, modality: str, **kwargs) -> None:
        self.modality = modality
        message = f"Adapter for modality '{modality}' is already registered. Use overwrite=True to replace."
        super().__init__(message, component_name=modality, **kwargs)
        self.error_code = self.ERROR_CODE


class AdapterNotFoundException(RegistryException):
    """Raised when looking up an unregistered modality key."""

    ERROR_CODE = "REG-102"

    def __init__(self, modality: str, supported_modalities: Optional[list] = None, **kwargs) -> None:
        self.modality = modality
        self.supported_modalities = supported_modalities or []
        message = (
            f"No adapter registered for modality '{modality}'. "
            f"Supported modalities: {self.supported_modalities}"
        )
        super().__init__(message, component_name=modality, **kwargs)
        self.error_code = self.ERROR_CODE


class AdapterLoadException(RegistryException):
    """Raised when an adapter fails to load model weights or initialize."""

    ERROR_CODE = "REG-103"

    def __init__(self, modality: str, cause: Optional[Exception] = None, **kwargs) -> None:
        self.modality = modality
        message = f"Failed to load model adapter for modality '{modality}': {cause}"
        super().__init__(message, component_name=modality, cause=cause, **kwargs)
        self.error_code = self.ERROR_CODE


class AdapterCacheException(RegistryException):
    """Raised on cache eviction or invalidation errors."""

    ERROR_CODE = "REG-104"

    def __init__(self, message: str = "Adapter cache operation failed.", **kwargs) -> None:
        super().__init__(message, **kwargs)
        self.error_code = self.ERROR_CODE
