"""
Medical AI Engine — PipelineStage Interface

Defines BasePipelineStage abstract contract for all execution pipeline stages.
Every stage exposes: validate(), process(), rollback(), cleanup().
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medical_ai_engine.core.engine_context import EngineContext


class BasePipelineStage(ABC):
    """
    Abstract Base Class for all pipeline execution stages.

    Lifecycle methods:
        validate(ctx)  → Pre-condition validation (raises if invalid)
        process(ctx)   → Main stage execution (mutates ctx in place)
        rollback(ctx)  → Undo mutations on failure
        cleanup(ctx)   → Release resources held by stage
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable stage name for logging and diagnostics."""
        pass

    def validate(self, ctx: EngineContext) -> None:
        """
        Pre-condition validation before process() is invoked.
        Default: no-op. Override when stage has input requirements.
        """
        pass

    @abstractmethod
    def process(self, ctx: EngineContext) -> None:
        """
        Main execution logic for the stage. Mutates ctx in place.

        Args:
            ctx: EngineContext carrying request state and intermediate outputs.
        """
        pass

    def rollback(self, ctx: EngineContext) -> None:
        """
        Undo partial mutations if process() raises an exception.
        Default: no-op. Override if stage mutates external resources.
        """
        pass

    def cleanup(self, ctx: EngineContext) -> None:
        """
        Teardown and resource cleanup. Called in finally block.
        Default: no-op. Override if stage holds temporary files/handles.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
