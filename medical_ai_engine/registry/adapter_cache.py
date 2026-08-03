"""
Medical AI Engine — Adapter Cache

Thread-safe memory cache for loaded IPredictorAdapter instances.
Supports cache lookup, insertion, invalidation, reload, and atomic hot-swapping.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional

from medical_ai_engine.interfaces.predictor_adapter import IPredictorAdapter
from medical_ai_engine.registry.registry_exceptions import AdapterCacheException


class AdapterCache:
    """
    Thread-safe in-memory cache for loaded model adapters.
    """

    def __init__(self) -> None:
        self._cache: Dict[str, IPredictorAdapter] = {}
        self._lock = threading.RLock()

    def get(self, modality: str) -> Optional[IPredictorAdapter]:
        """
        Get cached adapter instance for modality, or None if not cached.
        """
        key = modality.lower()
        with self._lock:
            return self._cache.get(key)

    def put(self, modality: str, adapter: IPredictorAdapter) -> None:
        """
        Put a loaded adapter instance into cache.
        """
        key = modality.lower()
        with self._lock:
            self._cache[key] = adapter

    def contains(self, modality: str) -> bool:
        """Returns True if cache contains loaded adapter for modality."""
        key = modality.lower()
        with self._lock:
            return key in self._cache and self._cache[key].is_loaded

    def invalidate(self, modality: Optional[str] = None) -> None:
        """
        Evict adapter from cache. If modality is None, clears entire cache.
        Calls adapter.shutdown() on evicted adapters to release RAM/VRAM.
        """
        with self._lock:
            if modality is None:
                for key, adapter in list(self._cache.items()):
                    try:
                        adapter.shutdown()
                    except Exception:
                        pass
                self._cache.clear()
            else:
                key = modality.lower()
                adapter = self._cache.pop(key, None)
                if adapter is not None:
                    try:
                        adapter.shutdown()
                    except Exception:
                        pass

    def cached_modalities(self) -> List[str]:
        """Return list of currently cached modality keys."""
        with self._lock:
            return [k for k, v in self._cache.items() if v.is_loaded]

    def size(self) -> int:
        """Return count of loaded adapters in cache."""
        with self._lock:
            return len(self.cached_modalities())
