"""
strategy_manager.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Strategy selection, fallback resolution, and execution invocation manager.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from medical_ai_engine.anomaly.interfaces.detection_strategy_interface import (
    IDetectionStrategy,
)
from .execution_metrics import ExecutionMetrics
from .framework_configuration import FrameworkConfiguration
from .framework_exceptions import StrategyExecutionError

logger = logging.getLogger(__name__)


class StrategyManager:
    """
    Manages selection, preference ordering, and safe invocation of detection strategies.

    Planned Strategy Framework Support (Module 16.2+)
    -------------------------------------------------
    - confidence_threshold  — Baseline confidence thresholding strategy
    - mahalanobis           — Mahalanobis distance OOD strategy
    - autoencoder           — Autoencoder reconstruction error strategy
    - deep_svdd             — Deep SVDD one-class strategy
    - openmax               — OpenMax open-set probability strategy
    - energy_based          — Energy-based score strategy
    """

    STRATEGY_PREFERENCE_ORDER = [
        "mahalanobis",
        "autoencoder",
        "deep_svdd",
        "openmax",
        "energy_based",
        "confidence_threshold",
    ]

    def __init__(
        self,
        registry: AnomalyRegistry,
        config: Optional[FrameworkConfiguration] = None,
        metrics: Optional[ExecutionMetrics] = None,
    ) -> None:
        self._registry = registry
        self._config = config or FrameworkConfiguration.default()
        self._metrics = metrics or ExecutionMetrics()

    def select_strategy(
        self, requested_key: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[IDetectionStrategy]]:
        """
        Select the requested strategy or fall back to preference order.
        """
        if requested_key and self._registry.is_registered(requested_key):
            return requested_key, self._registry.get_strategy(requested_key)

        # Fallback to preference order
        for key in self.STRATEGY_PREFERENCE_ORDER:
            if self._registry.is_registered(key):
                return key, self._registry.get_strategy(key)

        # Fallback to any registered strategy
        available = self._registry.list_strategies()
        if available:
            first_key = available[0]
            return first_key, self._registry.get_strategy(first_key)

        return None, None

    def execute_strategy(
        self, strategy: IDetectionStrategy, context: AnomalyContext
    ) -> bool:
        """
        Execute a strategy with timing, exception handling, and metrics tracking.

        Parameters
        ----------
        strategy : IDetectionStrategy
            The strategy to execute.
        context : AnomalyContext
            Shared execution context.

        Returns
        -------
        bool
            True if an anomaly was detected.
        """
        t0 = time.perf_counter()
        key = strategy.strategy_key
        is_anomaly = False
        success = False

        try:
            is_anomaly = strategy.detect(context)
            success = True
        except Exception as exc:
            logger.exception("[StrategyManager] Strategy '%s' execution failed: %s", key, exc)
            context.add_error("StrategyManager", f"Strategy '{key}' failed: {exc}")
            if not self._config.fail_silently:
                raise StrategyExecutionError(f"Strategy '{key}' failed: {exc}") from exc
            is_anomaly = False
            success = False
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._metrics.record_strategy_run(
                strategy_key=key, latency_ms=elapsed_ms, success=success
            )

        return is_anomaly


__all__ = ["StrategyManager"]
