"""
Medical AI Engine — Logger Interface

Abstract contract for engine logging implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class IEngineLogger(ABC):
    """
    Abstract interface for engine loggers.
    Supports structured logging with request_id context.
    """

    @abstractmethod
    def debug(self, msg: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def info(self, msg: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def warning(self, msg: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def error(self, msg: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass

    @abstractmethod
    def critical(self, msg: str, request_id: Optional[str] = None, extra: Optional[Dict[str, Any]] = None) -> None:
        pass
