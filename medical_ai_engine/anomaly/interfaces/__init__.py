"""
__init__.py — Module 16.1: Anomaly Detection Interfaces
Medical AI Platform — Package exports for anomaly framework interfaces.
"""

from __future__ import annotations

from .anomaly_registry_interface import IAnomalyRegistry
from .detection_strategy_interface import IDetectionStrategy

__all__ = ["IDetectionStrategy", "IAnomalyRegistry"]
