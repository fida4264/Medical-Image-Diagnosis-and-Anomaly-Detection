"""
Medical AI Engine — CAM Plugin Interface

Module 15.2: CAM Registry & Plugin System

Defines `ICAMPlugin` — the full plugin contract that every CAM method class
must implement to be loadable by the registry.

Relationship to Module 15.1 `ICAMMethod`:
  - `ICAMMethod`   (15.1) — minimal activation-map contract used by the pipeline.
  - `ICAMPlugin`   (15.2) — extended plugin contract adding lifecycle callbacks,
    configuration injection, health, and version reporting.
  - Concrete implementations in Module 15.3+ inherit from BOTH.

This two-interface design keeps the pipeline (which only cares about activation
maps) decoupled from the registry infrastructure concerns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from medical_ai_engine.explainability.registry.cam_configuration import CAMPluginConfiguration
from medical_ai_engine.explainability.registry.cam_metadata import CAMPluginMetadata


class ICAMPlugin(ABC):
    """
    Full plugin interface for Class Activation Mapping methods.

    Every CAM implementation registered with `CAMProductionRegistry` MUST
    satisfy this interface. The registry validates all required methods at
    registration time via `CAMPluginValidator`.

    Lifecycle callbacks (invoked by CAMFactory):
        on_initialize() → on_ready() → [on_activate() ↔ on_deactivate()] → on_unload()
    """

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    @abstractmethod
    def plugin_key(self) -> str:
        """Unique registry key (e.g. 'gradcam'). Must be lowercase, no spaces."""

    @property
    @abstractmethod
    def plugin_version(self) -> str:
        """Semantic version string (e.g. '1.0.0')."""

    # ── Configuration ─────────────────────────────────────────────────────────

    @abstractmethod
    def configure(self, config: CAMPluginConfiguration) -> None:
        """
        Inject runtime configuration into the plugin.

        Called by CAMFactory immediately after instantiation.
        Implementations store config but must NOT perform heavy computation here.
        """

    # ── Core Algorithm ────────────────────────────────────────────────────────

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
        Compute and return a 2D activation/saliency map.

        Returns:
            np.ndarray of shape (H, W), dtype float32, values in [0.0, 1.0].
        """

    # ── Lifecycle Callbacks ───────────────────────────────────────────────────

    @abstractmethod
    def on_initialize(self) -> None:
        """
        Called by CAMFactory when transitioning REGISTERED → INITIALIZED.

        Use to load model hooks, check GPU availability, or pre-allocate buffers.
        Must be idempotent.
        """

    @abstractmethod
    def on_ready(self) -> None:
        """
        Called by CAMFactory when transitioning INITIALIZED → READY.

        Use to perform any warm-up computation.
        """

    @abstractmethod
    def on_activate(self) -> None:
        """
        Called by CAMFactory when transitioning READY → ACTIVE.

        Plugin is now serving real requests. Register model hooks here if deferred.
        """

    @abstractmethod
    def on_deactivate(self) -> None:
        """
        Called by CAMFactory when transitioning ACTIVE → READY.

        Release any resources held for the active request.
        """

    @abstractmethod
    def on_unload(self) -> None:
        """
        Called by CAMFactory when transitioning to UNLOADED.

        Release ALL resources: hooks, GPU buffers, caches.
        """

    # ── Introspection ─────────────────────────────────────────────────────────

    @abstractmethod
    def get_plugin_metadata(self) -> CAMPluginMetadata:
        """Return the metadata descriptor for this plugin."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """
        Return a health status dict for this plugin instance.

        Suggested keys: is_healthy, state, last_error, memory_mb, device.
        """

    @abstractmethod
    def requires_gradients(self) -> bool:
        """True if this method requires PyTorch autograd (backward pass)."""
