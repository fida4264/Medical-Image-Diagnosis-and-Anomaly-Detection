"""
anomaly_registry.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Thread-safe plugin registry for anomaly detection strategies.

Design
------
The AnomalyRegistry implements IAnomalyRegistry and manages the full lifecycle
of all registered IDetectionStrategy plugins:
  1. register()      — Add a strategy to the registry (idempotent by key).
  2. initialize_all() — Initialize all strategies with config.
  3. get_strategy()  — Thread-safe lookup.
  4. cleanup_all()   — Graceful shutdown cleanup.

Plugin Registration Pattern
----------------------------
    from medical_ai_engine.anomaly import AnomalyRegistry
    from my_strategies import CustomMahalanobisStrategy

    registry = AnomalyRegistry()
    registry.register(CustomMahalanobisStrategy())
    registry.initialize_all(config={"ood_distance_threshold": 3.5})
    strategy = registry.get_strategy("mahalanobis")
    is_anomaly = strategy.detect(context)
"""

from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional

from .anomaly_exceptions import RegistryError, StrategyNotFoundError
from .interfaces.anomaly_registry_interface import IAnomalyRegistry
from .interfaces.detection_strategy_interface import IDetectionStrategy

logger = logging.getLogger(__name__)


class AnomalyRegistry(IAnomalyRegistry):
    """
    Thread-safe production-grade registry for anomaly detection strategy plugins.

    Features
    --------
    - RLock-protected reads and writes for concurrent-safe strategy access.
    - Idempotent registration: re-registering the same key overwrites the existing entry.
    - Batch lifecycle management: initialize_all() / cleanup_all().
    - Health monitoring: health_report() across all registered strategies.
    """

    def __init__(self) -> None:
        self._strategies: Dict[str, IDetectionStrategy] = {}
        self._lock = threading.RLock()
        self._initialized: bool = False
        logger.info("[AnomalyRegistry] Initialized (empty registry).")

    # ── Registration ──────────────────────────────────────────────────────────

    def register(self, strategy: IDetectionStrategy) -> None:
        if not isinstance(strategy, IDetectionStrategy):
            raise RegistryError(
                f"Object of type '{type(strategy).__name__}' does not implement IDetectionStrategy."
            )
        with self._lock:
            key = strategy.strategy_key
            if key in self._strategies:
                logger.warning("[AnomalyRegistry] Overwriting existing strategy '%s'.", key)
            self._strategies[key] = strategy
            logger.info(
                "[AnomalyRegistry] Registered strategy '%s' v%s.",
                key, strategy.strategy_version,
            )

    def unregister(self, strategy_key: str) -> None:
        with self._lock:
            if strategy_key not in self._strategies:
                raise StrategyNotFoundError(
                    f"Cannot unregister: strategy '{strategy_key}' is not registered."
                )
            del self._strategies[strategy_key]
            logger.info("[AnomalyRegistry] Unregistered strategy '%s'.", strategy_key)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_strategy(self, strategy_key: str) -> IDetectionStrategy:
        with self._lock:
            strategy = self._strategies.get(strategy_key)
        if strategy is None:
            available = list(self._strategies.keys())
            raise StrategyNotFoundError(
                f"Strategy '{strategy_key}' is not registered. "
                f"Available: {available}",
                details={"requested": strategy_key, "available": available},
            )
        return strategy

    def list_strategies(self) -> List[str]:
        with self._lock:
            return list(self._strategies.keys())

    def is_registered(self, strategy_key: str) -> bool:
        with self._lock:
            return strategy_key in self._strategies

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def initialize_all(self, config: Optional[dict] = None) -> None:
        config = config or {}
        with self._lock:
            keys = list(self._strategies.keys())

        for key in keys:
            try:
                strategy = self.get_strategy(key)
                strategy.initialize(config)
                logger.info("[AnomalyRegistry] Initialized strategy '%s'.", key)
            except Exception as exc:
                logger.error(
                    "[AnomalyRegistry] Failed to initialize strategy '%s': %s", key, exc
                )

        self._initialized = True

    def cleanup_all(self) -> None:
        with self._lock:
            keys = list(self._strategies.keys())

        for key in keys:
            try:
                strategy = self.get_strategy(key)
                strategy.cleanup()
                logger.info("[AnomalyRegistry] Cleaned up strategy '%s'.", key)
            except Exception as exc:
                logger.error(
                    "[AnomalyRegistry] Cleanup failed for strategy '%s': %s", key, exc
                )

        self._initialized = False

    # ── Health ────────────────────────────────────────────────────────────────

    def health_report(self) -> Dict[str, Dict]:
        report = {}
        with self._lock:
            keys = list(self._strategies.keys())

        for key in keys:
            try:
                strategy = self.get_strategy(key)
                report[key] = strategy.health_check()
            except Exception as exc:
                report[key] = {"strategy_key": key, "status": "ERROR", "error": str(exc)}

        return report

    def __repr__(self) -> str:
        return (
            f"AnomalyRegistry("
            f"strategies={list(self._strategies.keys())}, "
            f"initialized={self._initialized})"
        )


__all__ = ["AnomalyRegistry"]
