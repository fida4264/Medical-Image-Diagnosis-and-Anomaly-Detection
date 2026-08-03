"""
plugin_manager.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Plugin validation, metadata, and lifecycle manager.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from medical_ai_engine.anomaly.interfaces.detection_strategy_interface import (
    IDetectionStrategy,
)
from .framework_configuration import FrameworkConfiguration
from .framework_exceptions import PluginValidationError

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Manages strategy plugin validation, registration into AnomalyRegistry, and lifecycle checks.
    """

    def __init__(
        self,
        registry: Optional[AnomalyRegistry] = None,
        config: Optional[FrameworkConfiguration] = None,
    ) -> None:
        self._registry = registry or AnomalyRegistry()
        self._config = config or FrameworkConfiguration.default()
        self._lock = threading.RLock()

    def validate_plugin(self, strategy: IDetectionStrategy) -> bool:
        """
        Validate that a strategy plugin conforms to platform requirements.

        Validation Checks
        -----------------
        1. Must implement IDetectionStrategy.
        2. strategy_key must be non-empty string.
        3. strategy_version must be non-empty string.
        4. get_metadata() must return a valid Dict containing required keys.
        """
        if not isinstance(strategy, IDetectionStrategy):
            raise PluginValidationError(
                f"Plugin of type '{type(strategy).__name__}' does not implement IDetectionStrategy."
            )

        if not strategy.strategy_key or not isinstance(strategy.strategy_key, str):
            raise PluginValidationError(
                f"Plugin '{type(strategy).__name__}' has invalid strategy_key: {strategy.strategy_key!r}"
            )

        if not strategy.strategy_version or not isinstance(strategy.strategy_version, str):
            raise PluginValidationError(
                f"Plugin '{strategy.strategy_key}' has invalid strategy_version: {strategy.strategy_version!r}"
            )

        try:
            meta = strategy.get_metadata()
            if not isinstance(meta, dict) or "key" not in meta:
                raise PluginValidationError(
                    f"Plugin '{strategy.strategy_key}' get_metadata() returned invalid metadata schema."
                )
        except Exception as exc:
            raise PluginValidationError(
                f"Plugin '{strategy.strategy_key}' get_metadata() failed: {exc}"
            ) from exc

        return True

    def register_plugin(self, strategy: IDetectionStrategy) -> None:
        with self._lock:
            if self._config.enable_plugin_validation:
                self.validate_plugin(strategy)
            self._registry.register(strategy)

    def get_plugin_metadata(self, strategy_key: str) -> Dict[str, Any]:
        strategy = self._registry.get_strategy(strategy_key)
        return strategy.get_metadata()

    def list_plugin_metadata(self) -> List[Dict[str, Any]]:
        keys = self._registry.list_strategies()
        return [self.get_plugin_metadata(k) for k in keys]

    def get_registry(self) -> AnomalyRegistry:
        return self._registry


__all__ = ["PluginManager"]
