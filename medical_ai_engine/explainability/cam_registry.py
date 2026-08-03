"""
Medical AI Engine — CAM Registry

Module 15.1: Explainability Framework Architecture

Dynamic thread-safe registry responsible for discovering, registering, and
providing Class Activation Mapping (CAM) method plugins.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, Type

from medical_ai_engine.explainability.base_cam_method import (
    BaseCAMMethod,
    GradCAMMethodStub,
    GradCAMPlusPlusMethodStub,
    LayerCAMMethodStub,
    ScoreCAMMethodStub,
    EigenCAMMethodStub,
    IntegratedGradientsMethodStub,
    GuidedBackpropMethodStub,
)

logger = logging.getLogger("CAMRegistry")


class CAMRegistry:
    """
    Thread-safe registry managing available explainability methods.
    Supports dynamic registration of new CAM plugins without modifying core code.
    """

    _instance: Optional[CAMRegistry] = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._registry: Dict[str, Type[BaseCAMMethod]] = {}
        self._instances: Dict[str, BaseCAMMethod] = {}
        self._auto_register_builtins()

    @classmethod
    def get_instance(cls) -> CAMRegistry:
        """Singleton accessor."""
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for test isolation)."""
        with cls._singleton_lock:
            cls._instance = None

    def _auto_register_builtins(self) -> None:
        """Register default built-in CAM method stubs."""
        builtins = [
            ("gradcam", GradCAMMethodStub),
            ("gradcam_plus_plus", GradCAMPlusPlusMethodStub),
            ("layercam", LayerCAMMethodStub),
            ("scorecam", ScoreCAMMethodStub),
            ("eigencam", EigenCAMMethodStub),
            ("integrated_gradients", IntegratedGradientsMethodStub),
            ("guided_backprop", GuidedBackpropMethodStub),
        ]
        for key, cls in builtins:
            self.register(key, cls, overwrite=True)

    def register(
        self,
        method_key: str,
        method_cls: Type[BaseCAMMethod],
        overwrite: bool = False,
    ) -> None:
        """
        Register a new CAM method plugin.

        Args:
            method_key: String key (e.g. 'gradcam').
            method_cls: Class inheriting from BaseCAMMethod.
            overwrite: If True, allow replacing an existing registration.
        """
        key = method_key.lower()
        with self._lock:
            if key in self._registry and not overwrite:
                raise ValueError(f"CAM method '{key}' is already registered.")
            self._registry[key] = method_cls
            self._instances.pop(key, None)
            logger.debug("[CAMRegistry] Registered CAM method '%s' -> %s", key, method_cls.__name__)

    def unregister(self, method_key: str) -> bool:
        """Unregister a CAM method."""
        key = method_key.lower()
        with self._lock:
            self._instances.pop(key, None)
            return self._registry.pop(key, None) is not None

    def get_method_class(self, method_key: str) -> Type[BaseCAMMethod]:
        """Get registered CAM method class."""
        key = method_key.lower()
        with self._lock:
            if key not in self._registry:
                raise KeyError(
                    f"CAM method '{key}' not registered. Available: {list(self._registry.keys())}"
                )
            return self._registry[key]

    def get_method_instance(self, method_key: str) -> BaseCAMMethod:
        """Get or instantiate cached singleton instance of a CAM method."""
        key = method_key.lower()
        with self._lock:
            if key not in self._instances:
                cls = self.get_method_class(key)
                self._instances[key] = cls()
            return self._instances[key]

    def is_registered(self, method_key: str) -> bool:
        """Check if a CAM method is registered."""
        return method_key.lower() in self._registry

    def list_methods(self) -> List[str]:
        """Return list of all registered CAM method keys."""
        with self._lock:
            return sorted(list(self._registry.keys()))

    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata snapshot for all registered methods."""
        with self._lock:
            meta = {}
            for key in self.list_methods():
                inst = self.get_method_instance(key)
                meta[key] = inst.get_metadata()
            return meta
