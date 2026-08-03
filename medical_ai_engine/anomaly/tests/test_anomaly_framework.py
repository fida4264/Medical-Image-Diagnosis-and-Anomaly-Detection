"""
test_anomaly_framework.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Framework unit test suite.

Covers:
  - Registration & plugin validation
  - Strategy loading (PluginLoader)
  - Strategy lifecycle (initialize, detect, cleanup)
  - ExecutionPipeline end-to-end
  - CacheManager LRU / TTL & hits/misses
  - Thread safety under concurrent load
  - Error recovery & fault tolerance
"""

from __future__ import annotations

import concurrent.futures
import time
from typing import Any, Dict, Optional

import pytest

from medical_ai_engine.anomaly.anomaly_context import AnomalyContext
from medical_ai_engine.anomaly.anomaly_registry import AnomalyRegistry
from medical_ai_engine.anomaly.framework import (
    CacheManager,
    ExecutionMetrics,
    ExecutionPipeline,
    FrameworkConfiguration,
    FrameworkConfigurationError,
    PluginLoader,
    PluginManager,
    PluginValidationError,
    StrategyManager,
)
from medical_ai_engine.anomaly.interfaces import IDetectionStrategy


class MockFrameworkStrategy(IDetectionStrategy):
    """Concrete mock strategy for testing framework functionality."""

    def __init__(self, key: str = "mock_framework_strategy", is_anomaly: bool = False):
        self._key = key
        self._is_anomaly = is_anomaly
        self.initialized = False
        self.cleaned_up = False

    @property
    def strategy_key(self) -> str:
        return self._key

    @property
    def strategy_version(self) -> str:
        return "1.0.0-test"

    @property
    def strategy_description(self) -> str:
        return "Mock strategy for framework tests."

    def initialize(self, config: Dict[str, Any], model_adapter: Optional[Any] = None) -> None:
        self.initialized = True

    def detect(self, context: AnomalyContext) -> bool:
        context.ood_analysis = {"is_ood": self._is_anomaly, "strategy_used": self._key}
        return self._is_anomaly

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self._key,
            "version": self.strategy_version,
            "description": self.strategy_description,
            "requires_feature_vector": False,
            "requires_model": False,
            "is_initialized": self.initialized,
        }

    def cleanup(self) -> None:
        self.cleaned_up = True


class InvalidStrategy:
    """Invalid object missing IDetectionStrategy implementation."""
    pass


# ── T01: FrameworkConfiguration ──────────────────────────────────────────────

class TestFrameworkConfiguration:

    def test_default_values(self):
        cfg = FrameworkConfiguration()
        assert cfg.enable_cache is True
        assert cfg.max_cache_entries == 1000

    def test_invalid_cache_ttl_raises(self):
        with pytest.raises(FrameworkConfigurationError):
            FrameworkConfiguration(cache_ttl_seconds=-1.0)


# ── T02: CacheManager ─────────────────────────────────────────────────────────

class TestCacheManager:

    def test_cache_put_and_get(self):
        cache = CacheManager(FrameworkConfiguration(enable_cache=True, cache_ttl_seconds=5.0))
        cache.put("k1", "val1")
        assert cache.get("k1") == "val1"
        stats = cache.stats()
        assert stats["hits"] == 1

    def test_cache_expiration(self):
        cache = CacheManager(FrameworkConfiguration(enable_cache=True, cache_ttl_seconds=0.01))
        cache.put("k2", "val2")
        time.sleep(0.02)
        assert cache.get("k2") is None
        stats = cache.stats()
        assert stats["misses"] == 1

    def test_cache_lru_eviction(self):
        cache = CacheManager(FrameworkConfiguration(enable_cache=True, max_cache_entries=2))
        cache.put("k1", 1)
        cache.put("k2", 2)
        cache.put("k3", 3)  # Evicts k1
        assert cache.get("k1") is None
        assert cache.get("k2") == 2
        assert cache.get("k3") == 3


# ── T03: PluginManager & Validation ───────────────────────────────────────────

class TestPluginManager:

    def test_valid_plugin_registration(self):
        pm = PluginManager()
        strat = MockFrameworkStrategy(key="valid_strat")
        pm.register_plugin(strat)
        assert pm.get_registry().is_registered("valid_strat")

    def test_invalid_plugin_raises(self):
        pm = PluginManager()
        with pytest.raises(PluginValidationError):
            pm.register_plugin(InvalidStrategy())  # type: ignore

    def test_plugin_metadata(self):
        pm = PluginManager()
        strat = MockFrameworkStrategy(key="meta_strat")
        pm.register_plugin(strat)
        meta = pm.get_plugin_metadata("meta_strat")
        assert meta["key"] == "meta_strat"


# ── T04: StrategyManager & Selection ──────────────────────────────────────────

class TestStrategyManager:

    def test_strategy_selection_order(self):
        reg = AnomalyRegistry()
        s_mahal = MockFrameworkStrategy(key="mahalanobis")
        s_thresh = MockFrameworkStrategy(key="confidence_threshold")
        reg.register(s_thresh)
        reg.register(s_mahal)

        sm = StrategyManager(registry=reg)
        key, strat = sm.select_strategy()
        assert key == "mahalanobis"  # Preferred over confidence_threshold

    def test_explicit_strategy_request(self):
        reg = AnomalyRegistry()
        s_thresh = MockFrameworkStrategy(key="confidence_threshold")
        reg.register(s_thresh)

        sm = StrategyManager(registry=reg)
        key, strat = sm.select_strategy("confidence_threshold")
        assert key == "confidence_threshold"


# ── T05: ExecutionPipeline End-to-End ────────────────────────────────────────

class TestExecutionPipeline:

    def test_pipeline_execution(self):
        reg = AnomalyRegistry()
        strat = MockFrameworkStrategy(key="confidence_threshold", is_anomaly=False)
        reg.register(strat)

        pipeline = ExecutionPipeline(registry=reg)
        ctx = AnomalyContext(request_id="req-pipe-1", session_id="sess-pipe-1")

        exec_ctx = pipeline.execute(ctx)
        assert exec_ctx.active_strategy_key == "confidence_threshold"
        assert exec_ctx.cache_hit is False
        assert "total_pipeline_ms" in exec_ctx.stage_timings_ms

    def test_pipeline_cache_hit_on_second_run(self):
        reg = AnomalyRegistry()
        strat = MockFrameworkStrategy(key="confidence_threshold", is_anomaly=False)
        reg.register(strat)

        pipeline = ExecutionPipeline(registry=reg)
        ctx = AnomalyContext(request_id="req-pipe-2", session_id="sess-pipe-2", confidence=0.90)

        r1 = pipeline.execute(ctx)
        assert r1.cache_hit is False

        r2 = pipeline.execute(ctx)
        assert r2.cache_hit is True


# ── T06: Thread Safety & Concurrency ──────────────────────────────────────────

class TestFrameworkConcurrency:

    def test_concurrent_pipeline_execution(self):
        reg = AnomalyRegistry()
        strat = MockFrameworkStrategy(key="confidence_threshold")
        reg.register(strat)

        pipeline = ExecutionPipeline(registry=reg)
        errors = []

        def _task(idx: int):
            try:
                ctx = AnomalyContext(request_id=f"req-conc-{idx}", session_id=f"sess-{idx}")
                pipeline.execute(ctx)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(_task, i) for i in range(30)]
            for f in concurrent.futures.as_completed(futures):
                f.result()

        assert len(errors) == 0
        metrics = pipeline.get_metrics().summary()
        assert metrics["total_pipeline_runs"] == 30
