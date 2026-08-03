"""
Medical AI Engine — Base Stage Implementation

Extends BasePipelineStage interface providing common logging and helper methods.
"""

from __future__ import annotations

import logging
from abc import ABC
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage


class AbstractPipelineStage(BasePipelineStage, ABC):
    """
    Common base class for all concrete pipeline stages.
    Provides standard logger and lifecycle hooks.
    """

    def __init__(self, stage_name: str) -> None:
        self._name = stage_name
        self.logger = logging.getLogger(f"PipelineStage.{stage_name}")

    @property
    def name(self) -> str:
        return self._name
