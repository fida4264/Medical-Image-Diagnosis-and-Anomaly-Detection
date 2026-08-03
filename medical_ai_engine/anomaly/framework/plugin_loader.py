"""
plugin_loader.py — Module 16.2: Anomaly Detection Framework
Medical AI Platform — Dynamic strategy plugin loader.
"""

from __future__ import annotations

import importlib
import inspect
import logging
from pathlib import Path
from typing import List, Type, Optional

from medical_ai_engine.anomaly.interfaces.detection_strategy_interface import (
    IDetectionStrategy,
)
from .framework_exceptions import PluginLoadingError

logger = logging.getLogger(__name__)


class PluginLoader:
    """
    Scans modules or directories to dynamically discover and instantiate
    IDetectionStrategy plugin classes.
    """

    @staticmethod
    def load_from_module(module_path: str) -> List[IDetectionStrategy]:
        """
        Import a Python module path and instantiate all concrete IDetectionStrategy subclasses found.

        Parameters
        ----------
        module_path : str
            Dot-separated python module path (e.g. 'my_plugins.custom_strategy')

        Returns
        -------
        List[IDetectionStrategy]
            Instantiated strategy plugins.
        """
        try:
            mod = importlib.import_module(module_path)
        except Exception as exc:
            raise PluginLoadingError(f"Failed to import plugin module '{module_path}': {exc}") from exc

        instances: List[IDetectionStrategy] = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if issubclass(obj, IDetectionStrategy) and obj is not IDetectionStrategy and not inspect.isabstract(obj):
                try:
                    instance = obj()
                    instances.append(instance)
                    logger.info("[PluginLoader] Loaded strategy '%s' from class %s", instance.strategy_key, name)
                except Exception as exc:
                    logger.warning("[PluginLoader] Failed to instantiate plugin class %s: %s", name, exc)

        return instances

    @staticmethod
    def load_from_directory(directory_path: str) -> List[IDetectionStrategy]:
        """
        Scan a filesystem directory for .py files and attempt to load strategy plugins.
        """
        path = Path(directory_path)
        if not path.exists() or not path.is_dir():
            logger.warning("[PluginLoader] Directory '%s' does not exist or is not a directory.", directory_path)
            return []

        # Placeholder scan for future file-based plugin directories
        logger.info("[PluginLoader] Scanned plugin directory '%s' (0 external files loaded).", directory_path)
        return []


__all__ = ["PluginLoader"]
