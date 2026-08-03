"""
Medical AI Engine — Pipeline Stage Factory

Factory for instantiating the default 9-stage prediction pipeline
or custom pipeline configurations.
"""

from __future__ import annotations

from typing import List, Optional
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage
from medical_ai_engine.pipeline.stages import (
    AdapterPredictionStage,
    ClinicalReportHookStage,
    ConfidenceAssessmentStage,
    ImageQualityAssessmentStage,
    InputValidationStage,
    LoggingStage,
    ModalityClassificationStage,
    ResponsePreparationStage,
    RoutingStage,
)


class PipelineStageFactory:
    """
    Factory for creating ordered lists of BasePipelineStage instances.
    """

    @staticmethod
    def create_default_pipeline() -> List[BasePipelineStage]:
        """
        Create the standard 9-stage execution pipeline in strict order:
          1. InputValidationStage
          2. ImageQualityAssessmentStage
          3. ModalityClassificationStage
          4. RoutingStage
          5. AdapterPredictionStage
          6. ConfidenceAssessmentStage
          7. ClinicalReportHookStage
          8. ResponsePreparationStage
          9. LoggingStage
        """
        return [
            InputValidationStage(),
            ImageQualityAssessmentStage(),
            ModalityClassificationStage(),
            RoutingStage(),
            AdapterPredictionStage(),
            ConfidenceAssessmentStage(),
            ClinicalReportHookStage(),
            ResponsePreparationStage(),
            LoggingStage(),
        ]
