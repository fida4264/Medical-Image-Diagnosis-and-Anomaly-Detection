"""
Medical AI Engine — Base CAM Plugin

Module 15.2: CAM Registry & Plugin System

`BaseCAMPlugin` is the production-grade abstract base class that ALL concrete
CAM implementations (Grad-CAM, Score-CAM, etc.) in Module 15.3+ will inherit.

It bridges Module 15.1 (`BaseCAMMethod`) and Module 15.2 (`ICAMPlugin`), so
the existing pipeline continues to work unchanged while the new registry system
gains full lifecycle management.

Design:
  BaseCAMMethod (Module 15.1)   ← backward-compatible pipeline interface
  ICAMPlugin    (Module 15.2)   ← registry lifecycle & config interface
  BaseCAMPlugin (THIS FILE)     ← concrete base wiring both contracts

Concrete implementations ONLY override:
  - `generate_activation_map()` with their algorithm
  - Optionally override lifecycle hooks if needed
"""

from __future__ import annotations

import logging
import time
from abc import abstractmethod
from typing import Any

import numpy as np

from medical_ai_engine.explainability.base_cam_method import BaseCAMMethod
from medical_ai_engine.explainability.interfaces.cam_plugin_interface import ICAMPlugin
from medical_ai_engine.explainability.registry.cam_configuration import CAMPluginConfiguration
from medical_ai_engine.explainability.registry.cam_metadata import (
    CAMPluginMetadata,
    PluginLifecycleState,
    PluginStatus,
)

logger = logging.getLogger("BaseCAMPlugin")


class BaseCAMPlugin(BaseCAMMethod, ICAMPlugin):
    """
    Production base class for all CAM algorithm implementations.

    Implements all ICAMPlugin lifecycle methods with safe no-op defaults
    so concrete classes only need to override what they care about.

    Inherits from BaseCAMMethod to preserve Module 15.1 pipeline compatibility.
    """

    def __init__(
        self,
        plugin_key: str = "",
        plugin_version: str = "0.15.3.0",
        requires_gradients_flag: bool = True,
        description: str = "",
    ) -> None:
        key = plugin_key or self.__class__.__name__.lower()
        super().__init__(name=key, requires_gradients=requires_gradients_flag)

        self._plugin_key = key
        self._plugin_version = plugin_version
        self._description = description
        self._config: CAMPluginConfiguration = CAMPluginConfiguration.default_for_method(plugin_key)
        self._lifecycle_state: PluginLifecycleState = PluginLifecycleState.REGISTERED
        self._last_error: str = ""
        self._initialize_time: float = 0.0

    # ── ICAMPlugin: Identity ──────────────────────────────────────────────────

    @property
    def plugin_key(self) -> str:
        return self._plugin_key

    @property
    def plugin_version(self) -> str:
        return self._plugin_version

    # ── ICAMPlugin: Configuration ─────────────────────────────────────────────

    def configure(self, config: CAMPluginConfiguration) -> None:
        """Inject runtime config. Called by CAMFactory after instantiation."""
        self._config = config
        logger.debug("[%s] Configuration injected: %s", self._plugin_key, config.to_dict())

    # ── ICAMPlugin: Lifecycle Callbacks ───────────────────────────────────────

    def on_initialize(self) -> None:
        """REGISTERED → INITIALIZED. Override to set up hooks or allocate resources."""
        self._initialize_time = time.time()
        logger.debug("[%s] on_initialize() called.", self._plugin_key)

    def on_ready(self) -> None:
        """INITIALIZED → READY. Override to perform warm-up."""
        logger.debug("[%s] on_ready() called.", self._plugin_key)

    def on_activate(self) -> None:
        """READY → ACTIVE. Override to register model hooks for a live request."""
        logger.debug("[%s] on_activate() called.", self._plugin_key)

    def on_deactivate(self) -> None:
        """ACTIVE → READY. Override to release per-request resources."""
        logger.debug("[%s] on_deactivate() called.", self._plugin_key)

    def on_unload(self) -> None:
        """→ UNLOADED. Override to release ALL resources (hooks, GPU buffers)."""
        logger.debug("[%s] on_unload() called.", self._plugin_key)

    # ── ICAMPlugin: Core Algorithm (stub) ────────────────────────────────────

    @abstractmethod
    def generate_activation_map(
        self,
        model: Any,
        target_layer: Any,
        input_tensor: Any,
        target_category: int | None = None,
        extra_params: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """
        Concrete CAM algorithm. Must return a 2D float32 ndarray in [0, 1].
        Implemented by Module 15.3+ subclasses.
        """

    # ── ICAMPlugin: Introspection ─────────────────────────────────────────────

    def get_plugin_metadata(self) -> CAMPluginMetadata:
        from medical_ai_engine.explainability.registry.cam_metadata import BUILTIN_PLUGIN_METADATA
        if self._plugin_key in BUILTIN_PLUGIN_METADATA:
            meta = BUILTIN_PLUGIN_METADATA[self._plugin_key]
            meta.lifecycle_state = self._lifecycle_state
            return meta
        return CAMPluginMetadata(
            method_name=self._plugin_key,
            version=self._plugin_version,
            description=self._description,
            requires_gradients=self._requires_gradients,
            lifecycle_state=self._lifecycle_state,
            status=PluginStatus.EXPERIMENTAL,
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "plugin_key": self._plugin_key,
            "version": self._plugin_version,
            "lifecycle_state": self._lifecycle_state.value,
            "is_healthy": self._lifecycle_state in (
                PluginLifecycleState.READY,
                PluginLifecycleState.ACTIVE,
            ),
            "last_error": self._last_error,
            "config_device": self._config.device,
            "requires_gradients": self._requires_gradients,
        }

    def requires_gradients(self) -> bool:  # type: ignore[override]
        return self._requires_gradients

    # ── Module 15.1 compatibility: get_metadata ───────────────────────────────

    def get_metadata(self) -> dict[str, Any]:
        """Bridge to Module 15.1 ICAMMethod.get_metadata() contract."""
        return {
            "method_name": self._plugin_key,
            "version": self._plugin_version,
            "requires_gradients": self._requires_gradients,
            "lifecycle_state": self._lifecycle_state.value,
            "is_stub": False,
            "description": self._description,
        }
