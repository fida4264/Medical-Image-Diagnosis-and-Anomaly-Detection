"""
Medical AI Engine — CAM Plugin Exceptions

Module 15.2: CAM Registry & Plugin System

Defines the complete exception hierarchy for the CAM plugin system.
All registry, loader, factory, and cache operations raise these typed
exceptions so callers can respond precisely without inspecting strings.

Exception Hierarchy:
    CAMException                  (base)
    ├── CAMRegistrationError      (during register/unregister)
    │   ├── CAMDuplicateError     (key already registered, no overwrite)
    │   └── CAMInvalidPluginError (class does not satisfy plugin contract)
    ├── CAMLoadError              (during lazy load)
    │   └── CAMModuleNotFoundError
    ├── CAMFactoryError           (during factory creation)
    │   └── CAMLifecycleError     (invalid lifecycle state transition)
    ├── CAMCacheError             (during cache operations)
    └── CAMValidationError        (metadata or config validation fails)
"""

from __future__ import annotations


class CAMException(Exception):
    """Root of the CAM plugin system exception hierarchy."""

    def __init__(self, message: str, plugin_key: str = "") -> None:
        super().__init__(message)
        self.plugin_key = plugin_key
        self.message = message

    def __repr__(self) -> str:  # pragma: no cover
        return f"{self.__class__.__name__}(plugin_key={self.plugin_key!r}, message={self.message!r})"


# ── Registration Exceptions ───────────────────────────────────────────────────

class CAMRegistrationError(CAMException):
    """Raised when a CAM plugin cannot be registered."""


class CAMDuplicateError(CAMRegistrationError):
    """Raised when attempting to register a key that already exists without overwrite=True."""

    def __init__(self, plugin_key: str) -> None:
        super().__init__(
            f"CAM plugin '{plugin_key}' is already registered. "
            "Use overwrite=True to replace an existing registration.",
            plugin_key=plugin_key,
        )


class CAMInvalidPluginError(CAMRegistrationError):
    """Raised when a plugin class does not satisfy the ICAMPlugin interface contract."""

    def __init__(self, plugin_key: str, reason: str) -> None:
        super().__init__(
            f"Plugin '{plugin_key}' failed interface validation: {reason}",
            plugin_key=plugin_key,
        )
        self.reason = reason


# ── Load Exceptions ───────────────────────────────────────────────────────────

class CAMLoadError(CAMException):
    """Raised when a CAM plugin cannot be lazily loaded from a module path."""


class CAMModuleNotFoundError(CAMLoadError):
    """Raised when the Python module containing the plugin class cannot be imported."""

    def __init__(self, plugin_key: str, module_path: str) -> None:
        super().__init__(
            f"Cannot load plugin '{plugin_key}': module '{module_path}' not found.",
            plugin_key=plugin_key,
        )
        self.module_path = module_path


# ── Factory Exceptions ────────────────────────────────────────────────────────

class CAMFactoryError(CAMException):
    """Raised when the factory cannot create or configure a plugin instance."""


class CAMLifecycleError(CAMFactoryError):
    """Raised when an invalid plugin lifecycle state transition is attempted."""

    def __init__(self, plugin_key: str, current: str, attempted: str) -> None:
        super().__init__(
            f"Plugin '{plugin_key}': cannot transition from '{current}' to '{attempted}'.",
            plugin_key=plugin_key,
        )
        self.current_state = current
        self.attempted_state = attempted


# ── Cache Exceptions ──────────────────────────────────────────────────────────

class CAMCacheError(CAMException):
    """Raised during cache read/write/invalidation failures."""


# ── Validation Exceptions ─────────────────────────────────────────────────────

class CAMValidationError(CAMException):
    """Raised when plugin metadata or configuration fails schema validation."""

    def __init__(self, plugin_key: str, field: str, reason: str) -> None:
        super().__init__(
            f"Validation failed for plugin '{plugin_key}' — field '{field}': {reason}",
            plugin_key=plugin_key,
        )
        self.field = field
        self.reason = reason


# ── Not Found Exception ───────────────────────────────────────────────────────

class CAMNotFoundError(CAMException):
    """Raised when a requested CAM plugin key is not found in the registry."""

    def __init__(self, plugin_key: str, available: list[str] | None = None) -> None:
        available_str = ", ".join(available) if available else "none"
        super().__init__(
            f"CAM plugin '{plugin_key}' is not registered. Available: [{available_str}].",
            plugin_key=plugin_key,
        )
        self.available = available or []
