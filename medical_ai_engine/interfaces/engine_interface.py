"""
Medical AI Engine — Engine Interface

Abstract base class defining the contract for MedicalAIEngine.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.execution_session import ExecutionSession
from medical_ai_engine.core.engine_state import EngineState


class IEngine(ABC):
    """
    Abstract interface for Medical AI Orchestration Engines.
    Declares lifecycle management, request execution, and monitoring methods.
    """

    @property
    @abstractmethod
    def state(self) -> EngineState:
        """Current engine lifecycle state."""
        pass

    @abstractmethod
    def initialize(self) -> None:
        """Initialize engine resources, configuration, logging, and health monitors."""
        pass

    @abstractmethod
    def execute(
        self,
        image_source: Union[str, Path, Any, bytes],
        manual_modality_override: Optional[str] = None,
        request_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> EngineContext:
        """
        Execute a prediction request through the engine lifecycle.

        Args:
            image_source: Image file path, PIL Image, raw bytes, or array.
            manual_modality_override: Optional explicit modality key.
            request_id: Optional UUID string.
            extra_metadata: Optional metadata dict.

        Returns:
            EngineContext populated after execution.
        """
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Gracefully shut down the engine, clean up threads, and update state."""
        pass

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """Return operational engine statistics dictionary."""
        pass

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Return current engine health check status."""
        pass

    @abstractmethod
    def get_active_sessions(self) -> List[ExecutionSession]:
        """Return list of currently active execution sessions."""
        pass
