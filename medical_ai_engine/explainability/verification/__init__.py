"""
__init__.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Package exports for verification and benchmarking.
"""

from __future__ import annotations

from .architecture_validator import ArchitectureValidator
from .benchmark_runner import BenchmarkRunner
from .consistency_checker import ConsistencyChecker
from .faithfulness_evaluator import FaithfulnessEvaluator
from .integration_validator import SystemIntegrationValidator
from .memory_profiler import MemoryProfiler
from .performance_profiler import PerformanceProfiler
from .production_validator import ProductionValidator
from .quality_evaluator import QualityEvaluator
from .stress_tester import StressTester
from .verification_configuration import VerificationConfiguration
from .verification_exceptions import (
    ArchitectureValidationError,
    BenchmarkExecutionError,
    ClinicalExplanationVerificationError,
    ExplainabilityVerificationError,
    GradCAMVerificationError,
    ProductionVerificationError,
    QualityEvaluationError,
    RegressionTestError,
    StressTestError,
    VisualizationVerificationError,
)
from .verification_metrics import CategoryScore, VerificationMetrics
from .verification_report import VerificationReportGenerator
from .verification_runner import VerificationRunner

__all__ = [
    "VerificationRunner",
    "VerificationConfiguration",
    "VerificationMetrics",
    "CategoryScore",
    "ArchitectureValidator",
    "SystemIntegrationValidator",
    "QualityEvaluator",
    "FaithfulnessEvaluator",
    "ConsistencyChecker",
    "PerformanceProfiler",
    "BenchmarkRunner",
    "MemoryProfiler",
    "StressTester",
    "ProductionValidator",
    "VerificationReportGenerator",
    "ExplainabilityVerificationError",
    "ArchitectureValidationError",
    "GradCAMVerificationError",
    "VisualizationVerificationError",
    "ClinicalExplanationVerificationError",
    "BenchmarkExecutionError",
    "StressTestError",
    "QualityEvaluationError",
    "RegressionTestError",
    "ProductionVerificationError",
]
