"""
Medical AI Engine — CAM Plugin Loader

Module 15.2: CAM Registry & Plugin System

`CAMPluginLoader` handles lazy, on-demand loading of CAM plugin classes
from Python module paths. The registry stores only the class reference
initially; the loader only imports the implementation module when the
factory is first asked to create an instance.

Responsibilities:
  - Dynamic import of plugin classes by dotted module path + class name
  - Validation that the loaded class satisfies ICAMPlugin
  - Module-level import caching to avoid duplicate imports
  - Clear error reporting via typed exceptions

Design Principles:
  - NO plugin algorithm code is imported until explicitly requested
  - Supports future hot-swap: the loader can re-import a module on reload
  - Thread-safe import guard using per-key locks
"""

from __future__ import annotations

import importlib
import logging
import threading
from typing import Any, Type

from medical_ai_engine.explainability.registry.cam_exceptions import (
    CAMLoadError,
    CAMModuleNotFoundError,
    CAMInvalidPluginError,
)

logger = logging.getLogger("CAMPluginLoader")


class CAMPluginLoader:
    """
    Lazy loader for CAM plugin classes.

    Usage:
        loader = CAMPluginLoader()

        # Load by module path (requires module to be importable)
        cls = loader.load("gradcam", "medical_ai_engine.explainability.methods.gradcam", "GradCAMMethod")

        # Load directly from an already-imported class reference
        cls = loader.load_class("gradcam", GradCAMMethod)
    """

    def __init__(self) -> None:
        self._loaded_classes: dict[str, Type[Any]] = {}
        self._module_cache: dict[str, Any] = {}
        self._load_lock = threading.RLock()
        self._per_key_locks: dict[str, threading.Lock] = {}

    def _get_key_lock(self, plugin_key: str) -> threading.Lock:
        with self._load_lock:
            if plugin_key not in self._per_key_locks:
                self._per_key_locks[plugin_key] = threading.Lock()
            return self._per_key_locks[plugin_key]

    # ── Primary Load Methods ──────────────────────────────────────────────────

    def load(
        self,
        plugin_key: str,
        module_path: str,
        class_name: str,
        validate: bool = True,
    ) -> Type[Any]:
        """
        Lazily import and return the plugin class from a dotted module path.

        Args:
            plugin_key:   Registry key (for error reporting).
            module_path:  Dotted Python module path (e.g. 'my_pkg.cam.gradcam').
            class_name:   Class name within the module (e.g. 'GradCAMMethod').
            validate:     If True, validate the class satisfies ICAMPlugin contract.

        Returns:
            The plugin class (not an instance).

        Raises:
            CAMModuleNotFoundError: If the module cannot be imported.
            CAMLoadError:           If the class name doesn't exist in the module.
            CAMInvalidPluginError:  If validate=True and the contract check fails.
        """
        key = plugin_key.lower()
        key_lock = self._get_key_lock(key)

        with key_lock:
            if key in self._loaded_classes:
                logger.debug("[CAMPluginLoader] Cache hit for '%s'.", key)
                return self._loaded_classes[key]

            logger.info("[CAMPluginLoader] Loading plugin '%s' from '%s.%s'.", key, module_path, class_name)

            # Import module
            try:
                if module_path in self._module_cache:
                    module = self._module_cache[module_path]
                else:
                    module = importlib.import_module(module_path)
                    self._module_cache[module_path] = module
            except ImportError as exc:
                raise CAMModuleNotFoundError(key, module_path) from exc
            except Exception as exc:
                raise CAMLoadError(
                    f"Unexpected error importing module '{module_path}' for plugin '{key}': {exc}",
                    plugin_key=key,
                ) from exc

            # Retrieve class
            if not hasattr(module, class_name):
                raise CAMLoadError(
                    f"Module '{module_path}' has no class '{class_name}' for plugin '{key}'.",
                    plugin_key=key,
                )
            cls = getattr(module, class_name)

            if validate:
                self._validate_plugin_class(key, cls)

            self._loaded_classes[key] = cls
            logger.info("[CAMPluginLoader] Plugin '%s' loaded successfully.", key)
            return cls

    def load_class(
        self,
        plugin_key: str,
        cls: Type[Any],
        validate: bool = True,
    ) -> Type[Any]:
        """
        Register an already-imported class directly (e.g. from a stub or inline definition).

        Args:
            plugin_key: Registry key.
            cls:        The class to register.
            validate:   If True, validate against ICAMPlugin contract.
        """
        key = plugin_key.lower()
        if validate:
            self._validate_plugin_class(key, cls)
        with self._load_lock:
            self._loaded_classes[key] = cls
        logger.debug("[CAMPluginLoader] Directly registered class '%s' for key '%s'.", cls.__name__, key)
        return cls

    # ── Validation ────────────────────────────────────────────────────────────

    def _validate_plugin_class(self, plugin_key: str, cls: Type[Any]) -> None:
        """
        Validate that a class satisfies the ICAMPlugin contract.

        Checks:
          1. It is a class (not a function or instance).
          2. It declares the required abstract methods.

        Full ABC enforcement is done at instantiation time by Python;
        this pre-check gives a cleaner error message at registration time.
        """
        if not isinstance(cls, type):
            raise CAMInvalidPluginError(
                plugin_key, reason=f"'{cls}' is not a class."
            )

        required_methods = [
            "generate_activation_map",
            "configure",
            "on_initialize",
            "on_ready",
            "on_activate",
            "on_deactivate",
            "on_unload",
            "get_plugin_metadata",
            "health_check",
        ]

        missing = [m for m in required_methods if not hasattr(cls, m)]
        if missing:
            raise CAMInvalidPluginError(
                plugin_key,
                reason=f"Missing required methods: {missing}. "
                       f"Ensure '{cls.__name__}' inherits from BaseCAMPlugin.",
            )

    # ── Cache Management ──────────────────────────────────────────────────────

    def is_loaded(self, plugin_key: str) -> bool:
        return plugin_key.lower() in self._loaded_classes

    def unload(self, plugin_key: str) -> bool:
        """Remove the cached class (forces re-import on next load). Returns True if removed."""
        key = plugin_key.lower()
        with self._load_lock:
            if key in self._loaded_classes:
                del self._loaded_classes[key]
                logger.debug("[CAMPluginLoader] Unloaded class cache for '%s'.", key)
                return True
            return False

    def reload(
        self,
        plugin_key: str,
        module_path: str,
        class_name: str,
    ) -> Type[Any]:
        """
        Hot-swap a plugin by unloading its cached class and re-importing the module.
        This supports future live-reload scenarios.
        """
        key = plugin_key.lower()
        self.unload(key)
        # Force module re-import
        with self._load_lock:
            self._module_cache.pop(module_path, None)
        logger.info("[CAMPluginLoader] Hot-swap reload triggered for '%s'.", key)
        return self.load(key, module_path, class_name)

    def list_loaded(self) -> list[str]:
        with self._load_lock:
            return sorted(self._loaded_classes.keys())

    def stats(self) -> dict:
        return {
            "loaded_count": len(self._loaded_classes),
            "loaded_keys": self.list_loaded(),
            "cached_modules": len(self._module_cache),
        }
