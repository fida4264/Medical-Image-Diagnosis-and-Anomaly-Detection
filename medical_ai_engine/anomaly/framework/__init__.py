"""
__init__.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Framework package exports.
"""

from __future__ import annotations

from .cache_manager import CacheManager
from .execution_context import FrameworkExecutionContext
from .execution_metrics import ExecutionMetrics, StrategyMetricsSnapshot
from .framework_configuration import FrameworkConfiguration
from .framework_exceptions import (
    AnomalyFrameworkError,
    CacheError,
    FrameworkConfigurationError,
    PipelineExecutionError,
    PluginLoadingError,
    PluginValidationError,
    StrategyExecutionError,
)
from .pipeline import ExecutionPipeline
from .plugin_loader import PluginLoader
from .plugin_manager import PluginManager
from .strategy_manager import StrategyManager

__all__ = [
    "ExecutionPipeline",
    "StrategyManager",
    "PluginManager",
    "PluginLoader",
    "CacheManager",
    "FrameworkExecutionContext",
    "ExecutionMetrics",
    "StrategyMetricsSnapshot",
    "FrameworkConfiguration",
    "AnomalyFrameworkError",
    "FrameworkConfigurationError",
    "PluginLoadingError",
    "PluginValidationError",
    "StrategyExecutionError",
    "PipelineExecutionError",
    "CacheError",
]
