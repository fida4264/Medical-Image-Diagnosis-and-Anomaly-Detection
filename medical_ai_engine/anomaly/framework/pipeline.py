"""
pipeline.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Execution Pipeline orchestrating pre/exec/post/cache steps.

Pipeline Steps
--------------
1. Build FrameworkExecutionContext wrapper.
2. Check CacheManager for cached result.
3. Select & execute active detection strategy via StrategyManager.
4. Update cache on success.
5. Record execution metrics.
6. Return enriched execution context.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from .cache_manager import CacheManager
from .execution_context import FrameworkExecutionContext
from .execution_metrics import ExecutionMetrics
from .framework_configuration import FrameworkConfiguration
from .strategy_manager import StrategyManager

logger = logging.getLogger(__name__)


class ExecutionPipeline:
    """
    Framework execution pipeline orchestrating pre-execution, cache lookup,
    strategy invocation, cache storage, and post-execution metrics collection.
    """

    def __init__(
        self,
        registry: Optional[AnomalyRegistry] = None,
        config: Optional[FrameworkConfiguration] = None,
        cache_manager: Optional[CacheManager] = None,
        metrics: Optional[ExecutionMetrics] = None,
    ) -> None:
        self._config = config or FrameworkConfiguration.default()
        self._registry = registry or AnomalyRegistry()
        self._cache = cache_manager or CacheManager(self._config)
        self._metrics = metrics or ExecutionMetrics()
        self._strategy_manager = StrategyManager(
            registry=self._registry,
            config=self._config,
            metrics=self._metrics,
        )

    def execute(
        self,
        context: AnomalyContext,
        requested_strategy_key: Optional[str] = None,
    ) -> FrameworkExecutionContext:
        """
        Run the framework execution pipeline for a given AnomalyContext.

        Parameters
        ----------
        context : AnomalyContext
            Shared execution context.
        requested_strategy_key : Optional[str]
            Optional key of preferred strategy.

        Returns
        -------
        FrameworkExecutionContext
            Pipeline execution context wrapper containing timing breakdowns and metadata.
        """
        exec_ctx = FrameworkExecutionContext(context=context)
        t0 = time.perf_counter()

        try:
            # Step 1: Strategy Selection
            strategy_key, strategy = self._strategy_manager.select_strategy(
                requested_strategy_key
            )
            exec_ctx.active_strategy_key = strategy_key or "none"

            if strategy is None:
                context.log("No strategy registered for execution.", stage="Pipeline")
                self._metrics.record_pipeline_run(success=True)
                return exec_ctx

            # Step 2: Cache Lookup
            cache_key = f"{context.request_id}:{context.modality}:{strategy_key}:{context.confidence:.2f}"
            cached_result = self._cache.get(cache_key)

            if cached_result is not None:
                exec_ctx.cache_hit = True
                context.log(f"Cache hit for strategy '{strategy_key}'", stage="Pipeline")
                self._metrics.record_strategy_run(
                    strategy_key=strategy_key,
                    latency_ms=exec_ctx.elapsed_ms(),
                    success=True,
                    cached=True,
                )
                self._metrics.record_pipeline_run(success=True)
                return exec_ctx

            # Step 3: Strategy Execution
            t_exec = time.perf_counter()
            is_anomaly = self._strategy_manager.execute_strategy(strategy, context)
            exec_dur_ms = (time.perf_counter() - t_exec) * 1000.0
            exec_ctx.record_stage_timing("strategy_execution_ms", exec_dur_ms)

            # Step 4: Cache Update
            self._cache.put(cache_key, is_anomaly)

            self._metrics.record_pipeline_run(success=True)

        except Exception as exc:
            logger.exception("[ExecutionPipeline] Pipeline execution failed: %s", exc)
            context.add_error("ExecutionPipeline", str(exc))
            self._metrics.record_pipeline_run(success=False)

        finally:
            total_ms = (time.perf_counter() - t0) * 1000.0
            exec_ctx.record_stage_timing("total_pipeline_ms", total_ms)

        return exec_ctx

    def get_metrics(self) -> ExecutionMetrics:
        return self._metrics

    def get_cache(self) -> CacheManager:
        return self._cache

    def get_strategy_manager(self) -> StrategyManager:
        return self._strategy_manager


__all__ = ["ExecutionPipeline"]
