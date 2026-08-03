"""
Medical AI Engine — Configuration Interface

Abstract contract for engine configuration providers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class IEngineConfiguration(ABC):
    """
    Abstract interface for engine configuration objects.
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""
        pass
