"""
Medical AI Engine — CAM Plugin Factory

Module 15.2: CAM Registry & Plugin System

`CAMPluginFactory` is responsible for the complete plugin instantiation workflow:

  1. Retrieve the plugin class from `CAMProductionRegistry`
  2. Check `CAMPluginCache` — return cached instance if valid
  3. Instantiate the plugin
  4. Inject `CAMPluginConfiguration`
  5. Drive the lifecycle FSM: REGISTERED → INITIALIZED → READY
  6. Store the ready instance in the cache
  7. Return the ready plugin

The factory is the ONLY place that creates plugin instances. No other code
should call `GradCAMMethod()` directly.

Lifecycle Management:
  - `create(key)` → drives to READY state
  - `activate(key, instance)` → transitions READY → ACTIVE
  - `deactivate(key, instance)` → transitions ACTIVE → READY
  - `destroy(key)` → drives to UNLOADED + invalidates cache
  - `reload(key)` → drives to RELOADED → READY
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from medical_ai_engine.explainability.registry.cam_cache import CAMPluginCache
from medical_ai_engine.explainability.registry.cam_configuration import CAMPluginConfiguration
from medical_ai_engine.explainability.registry.cam_exceptions import (
    CAMFactoryError,
    CAMLifecycleError,
    CAMNotFoundError,
)
from medical_ai_engine.explainability.registry.cam_metadata import PluginLifecycleState

logger = logging.getLogger("CAMPluginFactory")


class CAMPluginFactory:
    """
    Orchestrates plugin instantiation and lifecycle management.

    The factory does NOT import the registry module at class-definition time.
    It receives the registry at construction time (dependency injection) to
    avoid circular imports and to allow test doubles.

    Usage:
        registry = CAMProductionRegistry.get_instance()
        factory = CAMPluginFactory(registry=registry)

        plugin = factory.create("gradcam")              # READY
        factory.activate("gradcam", plugin)             # ACTIVE
        factory.deactivate("gradcam", plugin)           # READY
        factory.destroy("gradcam", plugin)              # UNLOADED
    """

    def __init__(
        self,
        registry: Any,
        cache: CAMPluginCache | None = None,
    ) -> None:
        self._registry = registry
        self._cache = cache or CAMPluginCache(max_capacity=7)
        self._lock = threading.RLock()
        self._creation_times: dict[str, float] = {}

    # ── Primary Factory Method ────────────────────────────────────────────────

    def create(
        self,
        plugin_key: str,
        config: CAMPluginConfiguration | None = None,
        use_cache: bool = True,
    ) -> Any:
        """
        Create (or retrieve from cache) a READY plugin instance.

        Args:
            plugin_key: Registry key (e.g. 'gradcam').
            config:     Optional configuration override. Default: method-specific defaults.
            use_cache:  If True and a valid cached instance exists, return it directly.

        Returns:
            A plugin instance in the READY (or ACTIVE) lifecycle state.

        Raises:
            CAMNotFoundError:  If plugin_key is not registered.
            CAMFactoryError:   If instantiation or lifecycle transition fails.
        """
        key = plugin_key.lower()

        if use_cache:
            cached = self._cache.get(key)
            if cached is not None:
                logger.debug("[CAMPluginFactory] Cache hit for '%s'.", key)
                self._registry.get_metadata(key).record_use()
                return cached

        return self._instantiate_and_initialize(key, config)

    def _instantiate_and_initialize(
        self,
        plugin_key: str,
        config: CAMPluginConfiguration | None,
    ) -> Any:
        """Internal: instantiate, configure, and drive to READY state."""
        with self._lock:
            t0 = time.perf_counter()

            # 1. Retrieve class from registry
            if not self._registry.is_registered(plugin_key):
                raise CAMNotFoundError(plugin_key, self._registry.list_keys())
            plugin_cls = self._registry.get_plugin_class(plugin_key)

            # 2. Instantiate
            try:
                instance = plugin_cls()
            except Exception as exc:
                raise CAMFactoryError(
                    f"Failed to instantiate plugin '{plugin_key}': {exc}",
                    plugin_key=plugin_key,
                ) from exc

            # 3. Inject configuration
            resolved_config = config or CAMPluginConfiguration.default_for_method(plugin_key)
            try:
                instance.configure(resolved_config)
            except Exception as exc:
                raise CAMFactoryError(
                    f"Configuration injection failed for '{plugin_key}': {exc}",
                    plugin_key=plugin_key,
                ) from exc

            # 4. Drive lifecycle: REGISTERED → INITIALIZED → READY
            self._drive_to_ready(plugin_key, instance)

            # 5. Record metadata
            self._registry.get_metadata(plugin_key).record_use()

            elapsed_ms = (time.perf_counter() - t0) * 1000
            self._creation_times[plugin_key] = elapsed_ms
            logger.info(
                "[CAMPluginFactory] Plugin '%s' is READY (%.1f ms).",
                plugin_key, elapsed_ms,
            )

            # 6. Cache and return
            self._cache.put(plugin_key, instance)
            return instance

    def _drive_to_ready(self, plugin_key: str, instance: Any) -> None:
        """Drive lifecycle: REGISTERED → INITIALIZED → READY."""
        try:
            self._registry.transition_state(plugin_key, PluginLifecycleState.INITIALIZED)
            instance.on_initialize()
            self._registry.transition_state(plugin_key, PluginLifecycleState.READY)
            instance.on_ready()
        except CAMLifecycleError:
            raise
        except Exception as exc:
            raise CAMFactoryError(
                f"Lifecycle initialization failed for '{plugin_key}': {exc}",
                plugin_key=plugin_key,
            ) from exc

    # ── Lifecycle Methods ─────────────────────────────────────────────────────

    def activate(self, plugin_key: str, instance: Any) -> None:
        """Transition READY → ACTIVE and call on_activate()."""
        key = plugin_key.lower()
        try:
            self._registry.transition_state(key, PluginLifecycleState.ACTIVE)
            instance.on_activate()
            logger.debug("[CAMPluginFactory] '%s' → ACTIVE.", key)
        except CAMLifecycleError:
            raise
        except Exception as exc:
            raise CAMFactoryError(
                f"Activation failed for '{key}': {exc}", plugin_key=key
            ) from exc

    def deactivate(self, plugin_key: str, instance: Any) -> None:
        """Transition ACTIVE → READY and call on_deactivate()."""
        key = plugin_key.lower()
        try:
            self._registry.transition_state(key, PluginLifecycleState.READY)
            instance.on_deactivate()
            logger.debug("[CAMPluginFactory] '%s' → READY (deactivated).", key)
        except CAMLifecycleError:
            raise
        except Exception as exc:
            raise CAMFactoryError(
                f"Deactivation failed for '{key}': {exc}", plugin_key=key
            ) from exc

    def destroy(self, plugin_key: str, instance: Any) -> None:
        """Transition → UNLOADED, call on_unload(), invalidate cache."""
        key = plugin_key.lower()
        try:
            self._registry.transition_state(key, PluginLifecycleState.UNLOADED)
            instance.on_unload()
            self._cache.invalidate(key)
            logger.info("[CAMPluginFactory] '%s' → UNLOADED.", key)
        except CAMLifecycleError:
            raise
        except Exception as exc:
            raise CAMFactoryError(
                f"Destroy failed for '{key}': {exc}", plugin_key=key
            ) from exc

    def reload(self, plugin_key: str, instance: Any) -> Any:
        """Drive UNLOADED → RELOADED → READY. Returns a fresh READY instance."""
        key = plugin_key.lower()
        try:
            self._registry.transition_state(key, PluginLifecycleState.RELOADED)
        except CAMLifecycleError:
            raise
        # Re-create from scratch (fresh config injection + lifecycle)
        self._cache.invalidate(key)
        return self._instantiate_and_initialize(key, config=None)

    # ── Introspection ─────────────────────────────────────────────────────────

    def creation_stats(self) -> dict[str, float]:
        """Return creation time (ms) per plugin key."""
        return dict(self._creation_times)

    def cache_stats(self) -> dict:
        return self._cache.stats()
