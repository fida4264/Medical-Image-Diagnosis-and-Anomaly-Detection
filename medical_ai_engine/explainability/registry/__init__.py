"""
Medical AI Engine — CAM Registry Sub-package

Module 15.2: CAM Registry & Plugin System

Public Exports:
    CAMProductionRegistry  — Primary singleton plugin registry
    CAMPluginFactory       — Plugin creation & lifecycle manager
    CAMPluginLoader        — Dynamic lazy module loader
    CAMPluginCache         — Thread-safe LRU instance cache
    CAMPluginMetadata      — Metadata descriptor dataclass
    CAMPluginConfiguration — Per-plugin runtime configuration
    PluginLifecycleState   — Lifecycle FSM states
    PluginStatus           — Deployment status enum
"""

from medical_ai_engine.explainability.registry.cam_exceptions import (
    CAMException,
    CAMRegistrationError,
    CAMDuplicateError,
    CAMInvalidPluginError,
    CAMLoadError,
    CAMModuleNotFoundError,
    CAMFactoryError,
    CAMLifecycleError,
    CAMCacheError,
    CAMValidationError,
    CAMNotFoundError,
)
from medical_ai_engine.explainability.registry.cam_metadata import (
    CAMPluginMetadata,
    PluginStatus,
    PluginLifecycleState,
    BUILTIN_PLUGIN_METADATA,
)
from medical_ai_engine.explainability.registry.cam_configuration import (
    CAMPluginConfiguration,
)
from medical_ai_engine.explainability.registry.cam_cache import (
    CAMPluginCache,
    CacheEntry,
)
from medical_ai_engine.explainability.registry.cam_loader import (
    CAMPluginLoader,
)
from medical_ai_engine.explainability.registry.cam_factory import (
    CAMPluginFactory,
)
from medical_ai_engine.explainability.registry.cam_registry import (
    CAMProductionRegistry,
)

__all__ = [
    # Main registry and factory
    "CAMProductionRegistry",
    "CAMPluginFactory",
    "CAMPluginLoader",
    "CAMPluginCache",
    "CacheEntry",
    # Metadata & Config
    "CAMPluginMetadata",
    "CAMPluginConfiguration",
    "PluginStatus",
    "PluginLifecycleState",
    "BUILTIN_PLUGIN_METADATA",
    # Exceptions
    "CAMException",
    "CAMRegistrationError",
    "CAMDuplicateError",
    "CAMInvalidPluginError",
    "CAMLoadError",
    "CAMModuleNotFoundError",
    "CAMFactoryError",
    "CAMLifecycleError",
    "CAMCacheError",
    "CAMValidationError",
    "CAMNotFoundError",
]
