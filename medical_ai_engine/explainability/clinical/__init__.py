"""
__init__.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Public exports for the clinical sub-package.
"""

from __future__ import annotations

from .clinical_configuration import (
    ClinicalConfiguration,
    DEFAULT_REGION_BINS,
    DEFAULT_SEVERITY_THRESHOLDS,
    SUPPORTED_MODALITIES,
)
from .clinical_context import ClinicalContext
from .clinical_exceptions import (
    ClinicalExplanationError,
    ClinicalContextError,
    ConfigurationError,
    FindingClassificationError,
    RecommendationEngineError,
    RegionDescriptionError,
    ReportBuildError,
    SeverityAssessmentError,
    TextGenerationError,
)
from .clinical_report_builder import ClinicalReportBuilder
from .clinical_text_generator import ClinicalTextGenerator
from .finding_classifier import FindingClassifier
from .recommendation_engine import RecommendationEngine
from .region_descriptor import RegionDescriptor
from .severity_assessor import SeverityAssessor


class ClinicalExplanationEngine:
    """
    Facade that orchestrates the full 5-stage clinical explanation pipeline:

    Stage 1 — FindingClassifier   : anatomy + pathology classification
    Stage 2 — SeverityAssessor    : composite severity scoring
    Stage 3 — RegionDescriptor    : spatial natural-language descriptions
    Stage 4 — ClinicalTextGenerator: clinical impression paragraph
    Stage 5 — RecommendationEngine : actionable recommendations
    Stage 6 — ClinicalReportBuilder: final structured report dict

    Usage
    -----
    ::

        engine = ClinicalExplanationEngine()
        ctx = ClinicalContext(
            modality="brain_mri",
            disease_label="glioma",
            confidence_score=0.91,
            important_regions=[...],
            visualization_metrics={"coverage_ratio": 0.32, ...},
        )
        report = engine.explain(ctx)
        print(report["clinical_impression"])
    """

    def __init__(self, config: ClinicalConfiguration | None = None) -> None:
        self._config = config or ClinicalConfiguration()
        self._finder    = FindingClassifier(self._config)
        self._assessor  = SeverityAssessor(self._config)
        self._describer = RegionDescriptor(self._config)
        self._generator = ClinicalTextGenerator(self._config)
        self._recommender = RecommendationEngine(self._config)
        self._builder   = ClinicalReportBuilder(self._config)

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(
        self,
        ctx: ClinicalContext,
        export_path: str | None = None,
    ) -> dict:
        """
        Execute the full clinical explanation pipeline and return the report dict.

        Parameters
        ----------
        ctx:
            Populated ClinicalContext (at minimum: modality, disease_label,
            confidence_score).
        export_path:
            Optional file path to write the JSON report to disk.

        Returns
        -------
        dict
            Structured clinical report.
        """
        self._finder.classify(ctx)
        self._assessor.assess(ctx)
        self._describer.describe(ctx)
        self._generator.generate(ctx)
        self._recommender.recommend(ctx)
        return self._builder.build(ctx, export_path=export_path)

    @property
    def config(self) -> ClinicalConfiguration:
        return self._config


__all__ = [
    # Facade
    "ClinicalExplanationEngine",
    # Configuration & context
    "ClinicalConfiguration",
    "ClinicalContext",
    "DEFAULT_SEVERITY_THRESHOLDS",
    "DEFAULT_REGION_BINS",
    "SUPPORTED_MODALITIES",
    # Pipeline components
    "FindingClassifier",
    "SeverityAssessor",
    "RegionDescriptor",
    "ClinicalTextGenerator",
    "RecommendationEngine",
    "ClinicalReportBuilder",
    # Exceptions
    "ClinicalExplanationError",
    "ClinicalContextError",
    "ConfigurationError",
    "FindingClassificationError",
    "RecommendationEngineError",
    "RegionDescriptionError",
    "ReportBuildError",
    "SeverityAssessmentError",
    "TextGenerationError",
]
