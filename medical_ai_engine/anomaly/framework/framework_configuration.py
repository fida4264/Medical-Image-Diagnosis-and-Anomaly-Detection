"""
framework_configuration.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Framework-level configuration dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .framework_exceptions import FrameworkConfigurationError


@dataclass
class FrameworkConfiguration:
    """
    Configuration options for the Anomaly Detection Execution Framework.

    Attributes
    ----------
    enable_cache : bool
        Enable caching of strategy execution results.
    cache_ttl_seconds : float
        Time-to-live for cached strategy results in seconds.
    max_cache_entries : int
        Maximum number of cached entries in LRU cache.
    enable_plugin_validation : bool
        Validate plugins strictly before registration.
    fail_silently : bool
        If True, pipeline execution failures fall back gracefully.
    strategy_execution_timeout_ms : float
        Timeout for individual strategy execution in milliseconds.
    plugin_dirs : List[str]
        Directories to scan for dynamic plugin loading.
    """

    enable_cache: bool = True
    cache_ttl_seconds: float = 300.0
    max_cache_entries: int = 1000
    enable_plugin_validation: bool = True
    fail_silently: bool = True
    strategy_execution_timeout_ms: float = 100.0
    plugin_dirs: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.cache_ttl_seconds < 0:
            raise FrameworkConfigurationError(
                f"cache_ttl_seconds must be >= 0; got {self.cache_ttl_seconds}"
            )
        if self.max_cache_entries <= 0:
            raise FrameworkConfigurationError(
                f"max_cache_entries must be > 0; got {self.max_cache_entries}"
            )
        if self.strategy_execution_timeout_ms <= 0:
            raise FrameworkConfigurationError(
                f"strategy_execution_timeout_ms must be > 0; got {self.strategy_execution_timeout_ms}"
            )

    @classmethod
    def default(cls) -> "FrameworkConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "FrameworkConfiguration":
        return cls(
            enable_cache=True,
            cache_ttl_seconds=10.0,
            max_cache_entries=100,
            enable_plugin_validation=True,
            fail_silently=True,
            strategy_execution_timeout_ms=500.0,
        )


__all__ = ["FrameworkConfiguration"]
