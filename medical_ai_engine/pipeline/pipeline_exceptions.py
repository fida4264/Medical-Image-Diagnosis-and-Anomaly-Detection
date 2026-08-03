"""
Medical AI Engine — Pipeline Exception Hierarchy

Defines exception types for pipeline execution and stage operations.
Inherits from EngineExecutionException (ENG-300 block).
"""

from __future__ import annotations

from typing import Optional
from medical_ai_engine.core.exceptions import EngineExecutionException


class PipelineExecutionException(EngineExecutionException):
    """
    Base exception raised when the PipelineExecutor encounters a stage failure.
    """

    ERROR_CODE = "ENG-320"

    def __init__(
        self,
        message: str = "Pipeline execution failed.",
        stage_name: Optional[str] = None,
        failed_stage_index: Optional[int] = None,
        cause: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        self.failed_stage_index = failed_stage_index
        super().__init__(message, stage_name=stage_name, cause=cause, **kwargs)
        self.error_code = self.ERROR_CODE


class StageValidationException(PipelineExecutionException):
    """
    Raised when pre-condition validation fails before stage process().
    """

    ERROR_CODE = "ENG-321"

    def __init__(
        self,
        message: str = "Stage validation failed.",
        stage_name: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(message, stage_name=stage_name, **kwargs)
        self.error_code = self.ERROR_CODE


class StageRollbackException(PipelineExecutionException):
    """
    Raised when an error occurs during stage rollback execution.
    """

    ERROR_CODE = "ENG-322"

    def __init__(
        self,
        message: str = "Stage rollback failed.",
        stage_name: Optional[str] = None,
        cause: Optional[Exception] = None,
        **kwargs,
    ) -> None:
        super().__init__(message, stage_name=stage_name, cause=cause, **kwargs)
        self.error_code = self.ERROR_CODE
