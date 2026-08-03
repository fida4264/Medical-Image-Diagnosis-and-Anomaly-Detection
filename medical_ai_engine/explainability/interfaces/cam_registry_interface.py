"""
Medical AI Engine — CAM Registry Interface

Module 15.2: CAM Registry & Plugin System

Defines `ICAMRegistry` — the abstract interface contract that every registry
implementation must satisfy. Decouples the engine from any concrete registry
class, enabling test doubles, alternative backends, or distributed registries
to be swapped in without modifying the engine core.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

from medical_ai_engine.explainability.registry.cam_metadata import (
    CAMPluginMetadata,
    PluginLifecycleState,
)


class ICAMRegistry(ABC):
    """
    Abstract interface for the CAM plugin registry.

    Concrete Implementation: `medical_ai_engine.explainability.registry.cam_registry.CAMProductionRegistry`
    """

    # ── Registration ──────────────────────────────────────────────────────────

    @abstractmethod
    def register(
        self,
        plugin_key: str,
        plugin_class: Type[Any],
        metadata: CAMPluginMetadata | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Register a CAM plugin class under the given key.

        Args:
            plugin_key:   Unique identifier (e.g. 'gradcam').
            plugin_class: Class implementing ICAMPlugin.
            metadata:     Optional pre-built metadata. Defaults auto-built from class.
            overwrite:    If True, replace an existing registration.

        Raises:
            CAMDuplicateError:     If key exists and overwrite=False.
            CAMInvalidPluginError: If plugin_class fails interface validation.
        """

    @abstractmethod
    def unregister(self, plugin_key: str) -> bool:
        """
        Remove a plugin from the registry.

        Returns:
            True if the plugin was registered and has been removed, False otherwise.
        """

    # ── Lookup ────────────────────────────────────────────────────────────────

    @abstractmethod
    def get_plugin_class(self, plugin_key: str) -> Type[Any]:
        """
        Return the class registered under plugin_key.

        Raises:
            CAMNotFoundError: If plugin_key is not registered.
        """

    @abstractmethod
    def get_metadata(self, plugin_key: str) -> CAMPluginMetadata:
        """
        Return the metadata record for a registered plugin.

        Raises:
            CAMNotFoundError: If plugin_key is not registered.
        """

    @abstractmethod
    def is_registered(self, plugin_key: str) -> bool:
        """Return True if a plugin is registered under the given key."""

    @abstractmethod
    def list_keys(self) -> list[str]:
        """Return a sorted list of all registered plugin keys."""

    @abstractmethod
    def list_all_metadata(self) -> dict[str, CAMPluginMetadata]:
        """Return a snapshot dict mapping key → metadata for all plugins."""

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    @abstractmethod
    def transition_state(
        self,
        plugin_key: str,
        target_state: PluginLifecycleState,
    ) -> None:
        """
        Advance a plugin's lifecycle FSM to the given state.

        Raises:
            CAMNotFoundError:  If plugin_key is not registered.
            CAMLifecycleError: If the transition is not valid.
        """

    # ── Discovery & Health ────────────────────────────────────────────────────

    @abstractmethod
    def filter_by_modality(self, modality: str) -> list[str]:
        """Return plugin keys that support the given imaging modality."""

    @abstractmethod
    def filter_by_state(self, state: PluginLifecycleState) -> list[str]:
        """Return plugin keys currently in the given lifecycle state."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """
        Return a structured health report for all registered plugins.

        Schema:
            {
              "total": int,
              "ready": int,
              "unloaded": int,
              "plugins": { plugin_key: { "state": str, "use_count": int, ... } }
            }
        """

    # ── Reset ─────────────────────────────────────────────────────────────────

    @abstractmethod
    def reset(self) -> None:
        """Clear all registrations. Intended for test isolation only."""
