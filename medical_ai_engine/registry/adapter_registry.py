"""
Medical AI Engine — Adapter Registry

Stores class registrations and metadata for all available disease prediction adapters.
Thread-safe.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Type

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter
from medical_ai_engine.registry.adapter_metadata import AdapterMetadata
from medical_ai_engine.registry.registry_exceptions import (
    AdapterNotFoundException,
    DuplicateAdapterRegistrationException,
)


class AdapterRegistry:
    """
    Thread-safe registry for mapping modality keys to IPredictorAdapter classes and AdapterMetadata.
    """

    def __init__(self) -> None:
        self._classes: Dict[str, Type[IPredictorAdapter]] = {}
        self._metadata: Dict[str, AdapterMetadata] = {}
        self._lock = threading.RLock()

    def register(
        self,
        modality: str,
        adapter_cls: Type[IPredictorAdapter],
        metadata: Optional[AdapterMetadata] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register an adapter class for a modality key.

        Args:
            modality: Modality key string (e.g. 'skin').
            adapter_cls: Class implementing IPredictorAdapter.
            metadata: Optional AdapterMetadata object.
            overwrite: If False, raises DuplicateAdapterRegistrationException if already registered.
        """
        key = modality.lower()
        with self._lock:
            if key in self._classes and not overwrite:
                raise DuplicateAdapterRegistrationException(modality=modality)

            self._classes[key] = adapter_cls
            self._metadata[key] = metadata or AdapterMetadata(modality=key)

    def get_class(self, modality: str) -> Type[IPredictorAdapter]:
        """
        Lookup registered adapter class for modality key.

        Raises:
            AdapterNotFoundException: If modality is not registered.
        """
        key = modality.lower()
        with self._lock:
            if key not in self._classes:
                raise AdapterNotFoundException(
                    modality=modality,
                    supported_modalities=self.list_modalities(),
                )
            return self._classes[key]

    def get_metadata(self, modality: str) -> AdapterMetadata:
        """
        Get metadata for registered modality.
        """
        key = modality.lower()
        with self._lock:
            if key not in self._metadata:
                raise AdapterNotFoundException(
                    modality=modality,
                    supported_modalities=self.list_modalities(),
                )
            return self._metadata[key]

    def set_metadata(self, modality: str, metadata: AdapterMetadata) -> None:
        """Update metadata for registered modality."""
        key = modality.lower()
        with self._lock:
            self._metadata[key] = metadata

    def is_registered(self, modality: str) -> bool:
        """Returns True if modality is registered."""
        key = modality.lower()
        with self._lock:
            return key in self._classes

    def list_modalities(self) -> List[str]:
        """Return sorted list of all registered modality keys."""
        with self._lock:
            return sorted(list(self._classes.keys()))

    def unregister(self, modality: str) -> None:
        """Unregister a modality key."""
        key = modality.lower()
        with self._lock:
            self._classes.pop(key, None)
            self._metadata.pop(key, None)
