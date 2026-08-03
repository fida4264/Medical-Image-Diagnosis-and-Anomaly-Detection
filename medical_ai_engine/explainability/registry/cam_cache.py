"""
Medical AI Engine — CAM Plugin Cache

Module 15.2: CAM Registry & Plugin System

Thread-safe LRU cache for instantiated CAM plugin objects.

Responsibilities:
  - Cache plugin instances to avoid repeated instantiation
  - Enforce a configurable maximum capacity (evict LRU on overflow)
  - Support per-key TTL expiry (time-to-live)
  - Support manual invalidation by key or full flush
  - Provide cache statistics for health reporting

Cache Policy: LRU (Least Recently Used) with optional TTL per entry.

Thread Safety: All public methods are protected by a single RLock.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from medical_ai_engine.explainability.registry.cam_exceptions import CAMCacheError

logger = logging.getLogger("CAMPluginCache")


@dataclass
class CacheEntry:
    """One entry in the LRU cache."""
    plugin_key: str
    instance: Any
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    ttl_seconds: float = 0.0   # 0.0 → never expires

    def is_expired(self) -> bool:
        if self.ttl_seconds <= 0.0:
            return False
        return (time.time() - self.created_at) > self.ttl_seconds

    def touch(self) -> None:
        self.last_accessed = time.time()
        self.access_count += 1


class CAMPluginCache:
    """
    Thread-safe LRU cache for CAM plugin instances.

    Usage:
        cache = CAMPluginCache(max_capacity=7, default_ttl=0.0)
        cache.put("gradcam", plugin_instance)
        plugin = cache.get("gradcam")   # None if not cached / expired
        cache.invalidate("gradcam")
        cache.flush()
    """

    def __init__(
        self,
        max_capacity: int = 7,
        default_ttl: float = 0.0,
    ) -> None:
        if max_capacity < 1:
            raise CAMCacheError("CAMPluginCache max_capacity must be ≥ 1.", plugin_key="")
        self._max_capacity = max_capacity
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "invalidations": 0,
            "flushes": 0,
        }

    # ── Core Operations ───────────────────────────────────────────────────────

    def get(self, plugin_key: str) -> Any | None:
        """
        Retrieve a cached plugin instance.

        Returns:
            The cached instance, or None if not cached / expired.
        """
        key = plugin_key.lower()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._stats["misses"] += 1
                return None
            if entry.is_expired():
                logger.debug("[CAMPluginCache] Entry '%s' expired — evicting.", key)
                del self._store[key]
                self._stats["evictions"] += 1
                self._stats["misses"] += 1
                return None
            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.touch()
            self._stats["hits"] += 1
            return entry.instance

    def put(
        self,
        plugin_key: str,
        instance: Any,
        ttl_seconds: float | None = None,
    ) -> None:
        """
        Store a plugin instance in the cache.

        If the cache is at capacity, the LRU entry is evicted first.
        """
        key = plugin_key.lower()
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                # Refresh existing entry
                self._store.move_to_end(key)
                self._store[key].instance = instance
                self._store[key].touch()
                return
            if len(self._store) >= self._max_capacity:
                evicted_key, _ = self._store.popitem(last=False)
                self._stats["evictions"] += 1
                logger.debug("[CAMPluginCache] LRU eviction: '%s'.", evicted_key)
            entry = CacheEntry(
                plugin_key=key,
                instance=instance,
                ttl_seconds=ttl,
            )
            self._store[key] = entry
            logger.debug("[CAMPluginCache] Cached plugin '%s' (TTL=%.1fs).", key, ttl)

    def invalidate(self, plugin_key: str) -> bool:
        """Remove a single entry. Returns True if it existed."""
        key = plugin_key.lower()
        with self._lock:
            if key in self._store:
                del self._store[key]
                self._stats["invalidations"] += 1
                logger.debug("[CAMPluginCache] Invalidated '%s'.", key)
                return True
            return False

    def flush(self) -> int:
        """Remove all entries. Returns the number of entries removed."""
        with self._lock:
            count = len(self._store)
            self._store.clear()
            self._stats["flushes"] += 1
            logger.debug("[CAMPluginCache] Full flush — %d entries removed.", count)
            return count

    def contains(self, plugin_key: str) -> bool:
        """True if the key is cached and not expired."""
        return self.get(plugin_key) is not None

    def evict_expired(self) -> int:
        """Sweep the cache and evict all expired entries. Returns count removed."""
        with self._lock:
            expired_keys = [k for k, e in self._store.items() if e.is_expired()]
            for k in expired_keys:
                del self._store[k]
                self._stats["evictions"] += 1
            return len(expired_keys)

    # ── Introspection ─────────────────────────────────────────────────────────

    def size(self) -> int:
        with self._lock:
            return len(self._store)

    def capacity(self) -> int:
        return self._max_capacity

    def stats(self) -> dict[str, int | float]:
        with self._lock:
            hit_rate = (
                self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                if (self._stats["hits"] + self._stats["misses"]) > 0
                else 0.0
            )
            return {
                **self._stats,
                "current_size": len(self._store),
                "capacity": self._max_capacity,
                "hit_rate": round(hit_rate, 4),
            }

    def snapshot(self) -> list[dict]:
        """Return a list of dicts describing each cached entry (for health reports)."""
        with self._lock:
            result = []
            for k, e in self._store.items():
                result.append({
                    "plugin_key": k,
                    "access_count": e.access_count,
                    "last_accessed": e.last_accessed,
                    "is_expired": e.is_expired(),
                })
            return result
