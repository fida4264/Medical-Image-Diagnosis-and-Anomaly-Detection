"""
cache_manager.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Thread-safe LRU / TTL cache manager for strategy results.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .framework_configuration import FrameworkConfiguration

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float
    ttl_seconds: float

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > self.ttl_seconds


class CacheManager:
    """
    Thread-safe LRU/TTL cache manager for caching anomaly strategy outcomes.

    Cache Key Formula
    -----------------
    `hash(request_id + modality + strategy_key + confidence)`
    """

    def __init__(self, config: Optional[FrameworkConfiguration] = None) -> None:
        self._config = config or FrameworkConfiguration.default()
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def get(self, cache_key: str) -> Optional[Any]:
        if not self._config.enable_cache:
            return None

        with self._lock:
            if cache_key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[cache_key]
            if entry.is_expired():
                del self._cache[cache_key]
                self._misses += 1
                return None

            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            self._hits += 1
            return entry.value

    def put(self, cache_key: str, value: Any, ttl_seconds: Optional[float] = None) -> None:
        if not self._config.enable_cache:
            return

        ttl = ttl_seconds if ttl_seconds is not None else self._config.cache_ttl_seconds

        with self._lock:
            # Evict if full
            if len(self._cache) >= self._config.max_cache_entries:
                self._cache.popitem(last=False)

            entry = CacheEntry(
                key=cache_key,
                value=value,
                created_at=time.time(),
                ttl_seconds=ttl,
            )
            self._cache[cache_key] = entry
            self._cache.move_to_end(cache_key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total) if total > 0 else 0.0
            return {
                "size": len(self._cache),
                "max_entries": self._config.max_cache_entries,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
            }


__all__ = ["CacheManager"]
