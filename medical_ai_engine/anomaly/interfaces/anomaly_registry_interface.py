"""
anomaly_registry_interface.py — Module 16.1: Unknown Disease & Anomaly Detection Architecture
Medical AI Platform — Abstract interface for the Anomaly Strategy Registry.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from .detection_strategy_interface import IDetectionStrategy


class IAnomalyRegistry(ABC):
    """
    Abstract interface contract for the AnomalyRegistry.

    The AnomalyRegistry is responsible for:
      - Strategy registration and discovery
      - Lifecycle management (initialize / cleanup)
      - Thread-safe strategy lookup
      - Health monitoring across all registered strategies
    """

    @abstractmethod
    def register(self, strategy: IDetectionStrategy) -> None:
        """
        Register a detection strategy.

        Parameters
        ----------
        strategy : IDetectionStrategy
            An uninitialized or pre-initialized strategy instance.
        """

    @abstractmethod
    def unregister(self, strategy_key: str) -> None:
        """
        Remove a strategy from the registry.

        Parameters
        ----------
        strategy_key : str
            Registry key of the strategy to remove.
        """

    @abstractmethod
    def get_strategy(self, strategy_key: str) -> IDetectionStrategy:
        """
        Retrieve a registered strategy by key.

        Raises
        ------
        StrategyNotFoundError
            If the strategy_key is not registered.
        """

    @abstractmethod
    def list_strategies(self) -> List[str]:
        """Return a list of all registered strategy keys."""

    @abstractmethod
    def is_registered(self, strategy_key: str) -> bool:
        """Return True if the given strategy key is registered."""

    @abstractmethod
    def health_report(self) -> Dict[str, Dict]:
        """
        Return a health status snapshot of all registered strategies.

        Returns
        -------
        Dict mapping strategy_key → health_status_dict.
        """

    @abstractmethod
    def initialize_all(self, config: Optional[dict] = None) -> None:
        """
        Initialize all registered strategies with the provided configuration.

        Called once at engine startup.
        """

    @abstractmethod
    def cleanup_all(self) -> None:
        """
        Call cleanup() on all registered strategies.
        Called during engine shutdown.
        """


__all__ = ["IAnomalyRegistry"]
