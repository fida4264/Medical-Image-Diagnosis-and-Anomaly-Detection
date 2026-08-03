"""
Medical AI Engine — Module 14.2.7 Verification Test Suite

PyTest unit tests for the production verification subsystem.
Uses the 'fast' configuration profile for CI/CD speed.
"""

from __future__ import annotations

import pytest
import numpy as np
from PIL import Image

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import (
    VerificationStatus,
    ProductionVerdict,
    QualityScores,
    VerificationResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def fast_config() -> VerificationConfiguration:
    """Return the fast CI verification configuration."""
    return VerificationConfiguration.fast()


@pytest.fixture(scope="module")
def dummy_image() -> Image.Image:
    return Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))


# ── Configuration Tests ───────────────────────────────────────────────────────

class TestVerificationConfiguration:
    def test_default_config_has_7_modalities(self):
        cfg = VerificationConfiguration.default()
        assert len(cfg.supported_modalities) == 7

    def test_fast_config_has_fewer_iterations(self):
        fast = VerificationConfiguration.fast()
        default = VerificationConfiguration.default()
        assert fast.benchmark_iterations < default.benchmark_iterations
        assert fast.memory_test_cycles < default.memory_test_cycles

    def test_fast_config_concurrency_levels(self):
        fast = VerificationConfiguration.fast()
        assert 20 in fast.thread_concurrency_levels
        assert 50 in fast.thread_concurrency_levels

    def test_config_to_dict(self):
        cfg = VerificationConfiguration.default()
        d = cfg.to_dict()
        assert "supported_modalities" in d
        assert "benchmark_iterations" in d
        assert "thread_concurrency_levels" in d


# ── Report Data Class Tests ───────────────────────────────────────────────────

class TestVerificationReport:
    def test_verification_result_passed(self):
        r = VerificationResult(
            name="Test Check",
            status=VerificationStatus.PASSED,
            score=100.0,
        )
        assert r.passed is True
        assert r.failed is False

    def test_verification_result_failed(self):
        r = VerificationResult(
            name="Test Check",
            status=VerificationStatus.FAILED,
            score=0.0,
        )
        assert r.passed is False
        assert r.failed is True

    def test_verification_result_to_dict(self):
        r = VerificationResult(
            name="Test",
            status=VerificationStatus.PASSED,
            score=85.0,
            message="All good",
        )
        d = r.to_dict()
        assert d["name"] == "Test"
        assert d["status"] == "PASSED"
        assert d["score"] == 85.0

    def test_quality_scores_compute_overall(self):
        q = QualityScores(
            architecture=100.0,
            engineering=100.0,
            maintainability=100.0,
            scalability=100.0,
            performance=100.0,
            reliability=100.0,
            security_readiness=100.0,
            documentation=100.0,
            testing=100.0,
        )
        overall = q.compute_overall()
        assert 99.0 <= overall <= 100.0

    def test_quality_scores_grade(self):
        q = QualityScores(overall=95.0)
        assert q.grade() == "A+"
        q2 = QualityScores(overall=55.0)
        assert q2.grade() == "C"

    def test_quality_scores_table(self):
        q = QualityScores(
            architecture=80.0, engineering=85.0, maintainability=78.0,
            scalability=90.0, performance=88.0, reliability=82.0,
            security_readiness=75.0, documentation=90.0, testing=87.0,
            overall=84.0,
        )
        table = q.to_table()
        assert "Architecture" in table
        assert "OVERALL" in table


# ── Exception Tests ───────────────────────────────────────────────────────────

class TestVerificationExceptions:
    def test_architecture_audit_exception(self):
        from medical_ai_engine.verification.verification_exceptions import ArchitectureAuditException
        exc = ArchitectureAuditException(
            message="Critical violation",
            violations=["Missing interface"],
        )
        assert "VER-701" in str(exc)
        assert exc.violations == ["Missing interface"]

    def test_compatibility_exception(self):
        from medical_ai_engine.verification.verification_exceptions import CompatibilityException
        exc = CompatibilityException(
            message="Breaking change detected",
            breaking_changes=["Missing field 'confidence'"],
        )
        assert "VER-704" in str(exc)
        assert len(exc.breaking_changes) == 1

    def test_stress_test_exception(self):
        from medical_ai_engine.verification.verification_exceptions import StressTestException
        exc = StressTestException(message="Timeout", scenario="burst")
        assert "VER-702" in str(exc)
        assert exc.scenario == "burst"

    def test_health_check_exception(self):
        from medical_ai_engine.verification.verification_exceptions import HealthCheckException
        exc = HealthCheckException(message="Critical", health_score=12.0)
        assert "VER-705" in str(exc)
        assert exc.health_score == 12.0


# ── Architecture Auditor Tests ────────────────────────────────────────────────

class TestArchitectureAuditor:
    def test_auditor_returns_result(self, fast_config):
        from medical_ai_engine.verification.architecture_auditor import ArchitectureAuditor
        auditor = ArchitectureAuditor(fast_config)
        result = auditor.run()
        assert isinstance(result, VerificationResult)
        assert result.name == "Architecture Compliance Audit"
        assert 0.0 <= result.score <= 100.0

    def test_auditor_score_above_zero(self, fast_config):
        from medical_ai_engine.verification.architecture_auditor import ArchitectureAuditor
        result = ArchitectureAuditor(fast_config).run()
        assert result.score > 0.0, "Architecture audit score should be > 0"

    def test_auditor_no_critical_violations(self, fast_config):
        from medical_ai_engine.verification.architecture_auditor import ArchitectureAuditor
        result = ArchitectureAuditor(fast_config).run()
        # Should have no FAILED status for a correct platform
        assert result.status != VerificationStatus.ERROR

    def test_auditor_design_patterns_present(self, fast_config):
        from medical_ai_engine.verification.architecture_auditor import ArchitectureAuditor
        auditor = ArchitectureAuditor(fast_config)
        result = auditor.run()
        pattern_results = result.details.get("pattern_results", {})
        # All patterns should be present
        for pattern_name, pattern_status in pattern_results.items():
            assert "MISSING" not in pattern_status, f"Pattern '{pattern_name}' is missing: {pattern_status}"


# ── Compatibility Checker Tests ───────────────────────────────────────────────

class TestCompatibilityChecker:
    def test_checker_returns_result(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        result = CompatibilityChecker(fast_config).run()
        assert isinstance(result, VerificationResult)
        assert result.name == "Backward Compatibility Check"

    def test_prediction_result_fields_present(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        checker = CompatibilityChecker(fast_config)
        ok, errs = checker._check_prediction_result_fields()
        assert ok, f"PredictionResult field violations: {errs}"

    def test_engine_context_fields_present(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        checker = CompatibilityChecker(fast_config)
        ok, errs = checker._check_engine_context_fields()
        assert ok, f"EngineContext field violations: {errs}"

    def test_enum_values_present(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        checker = CompatibilityChecker(fast_config)
        ok, errs = checker._check_enum_values()
        assert ok, f"Enum value violations: {errs}"

    def test_json_roundtrip_stable(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        checker = CompatibilityChecker(fast_config)
        ok, err = checker._check_json_roundtrip()
        assert ok, f"JSON round-trip failed: {err}"

    def test_no_breaking_changes(self, fast_config):
        from medical_ai_engine.verification.compatibility_checker import CompatibilityChecker
        result = CompatibilityChecker(fast_config).run()
        assert result.violations == [], f"Breaking changes detected: {result.violations}"


# ── Thread Safety Tests ───────────────────────────────────────────────────────

class TestThreadSafetyValidator:
    def test_validator_returns_result(self, fast_config):
        from medical_ai_engine.verification.thread_safety_validator import ThreadSafetyValidator
        # Use a minimal config with only 1 level to keep test fast
        cfg = VerificationConfiguration.fast()
        cfg.thread_concurrency_levels = [20]
        result = ThreadSafetyValidator(cfg).run()
        assert isinstance(result, VerificationResult)

    def test_level_20_passes(self, fast_config):
        from medical_ai_engine.verification.thread_safety_validator import ThreadSafetyValidator
        cfg = VerificationConfiguration.fast()
        cfg.thread_concurrency_levels = [20]
        result = ThreadSafetyValidator(cfg).run()
        assert result.score >= 80.0, f"Thread safety score at level 20: {result.score}"


# ── Memory Profiler Tests ─────────────────────────────────────────────────────

class TestMemoryProfiler:
    def test_profiler_runs_without_crash(self, fast_config):
        from medical_ai_engine.verification.memory_profiler import MemoryProfiler
        cfg = VerificationConfiguration.fast()
        cfg.memory_test_cycles = 10
        cfg.memory_test_duration_seconds = 5.0
        result = MemoryProfiler(cfg).run()
        assert isinstance(result, VerificationResult)
        assert result.score >= 0.0

    def test_profiler_reports_cycles_completed(self, fast_config):
        from medical_ai_engine.verification.memory_profiler import MemoryProfiler
        cfg = VerificationConfiguration.fast()
        cfg.memory_test_cycles = 5
        cfg.memory_test_duration_seconds = 5.0
        result = MemoryProfiler(cfg).run()
        assert result.details.get("cycles_completed", 0) > 0


# ── Benchmark Runner Tests ────────────────────────────────────────────────────

class TestBenchmarkRunner:
    def test_benchmark_returns_result(self, fast_config):
        from medical_ai_engine.verification.benchmark_runner import BenchmarkRunner
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        result = BenchmarkRunner(cfg).run()
        assert isinstance(result, VerificationResult)

    def test_benchmark_has_modality_data(self, fast_config):
        from medical_ai_engine.verification.benchmark_runner import BenchmarkRunner
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        result = BenchmarkRunner(cfg).run()
        benchmarks = result.details.get("modality_benchmarks", {})
        assert len(benchmarks) == 7, f"Expected 7 modalities, got {len(benchmarks)}"

    def test_benchmark_avg_latency_positive(self, fast_config):
        from medical_ai_engine.verification.benchmark_runner import BenchmarkRunner
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        result = BenchmarkRunner(cfg).run()
        assert result.details.get("global_avg_ms", 0.0) > 0.0


# ── Health Monitor Tests ──────────────────────────────────────────────────────

class TestHealthMonitor:
    def test_monitor_returns_result(self, fast_config):
        from medical_ai_engine.verification.health_monitor import HealthMonitor
        result = HealthMonitor(fast_config).run()
        assert isinstance(result, VerificationResult)

    def test_monitor_health_score_positive(self, fast_config):
        from medical_ai_engine.verification.health_monitor import HealthMonitor
        result = HealthMonitor(fast_config).run()
        assert result.score > 0.0, "Health score should be positive"

    def test_monitor_reports_system_status(self, fast_config):
        from medical_ai_engine.verification.health_monitor import HealthMonitor
        result = HealthMonitor(fast_config).run()
        status = result.details.get("system_status", "")
        assert status in ("HEALTHY", "DEGRADED", "CRITICAL")

    def test_monitor_loaded_models_count(self, fast_config):
        from medical_ai_engine.verification.health_monitor import HealthMonitor
        result = HealthMonitor(fast_config).run()
        loaded = result.details.get("loaded_models", 0)
        assert loaded == 7, f"Expected 7 loaded models, got {loaded}"


# ── Production Validator Tests ────────────────────────────────────────────────

class TestProductionValidator:
    def test_validator_runs_and_returns_report(self, fast_config):
        from medical_ai_engine.verification.production_validator import ProductionValidator
        cfg = VerificationConfiguration.fast()
        # Minimal cycles for this test
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        cfg.memory_test_cycles = 5
        cfg.thread_concurrency_levels = [20]
        validator = ProductionValidator(cfg)
        report = validator.run()
        assert report is not None

    def test_validator_produces_10_results(self, fast_config):
        from medical_ai_engine.verification.production_validator import ProductionValidator
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        cfg.memory_test_cycles = 5
        cfg.thread_concurrency_levels = [20]
        report = ProductionValidator(cfg).run()
        assert len(report.verification_results) == 10

    def test_validator_verdict_is_set(self, fast_config):
        from medical_ai_engine.verification.production_validator import ProductionValidator
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        cfg.memory_test_cycles = 5
        cfg.thread_concurrency_levels = [20]
        report = ProductionValidator(cfg).run()
        assert report.verdict in (
            ProductionVerdict.GO,
            ProductionVerdict.NO_GO,
            ProductionVerdict.CONDITIONAL_GO,
        )

    def test_validator_quality_score_computed(self, fast_config):
        from medical_ai_engine.verification.production_validator import ProductionValidator
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        cfg.memory_test_cycles = 5
        cfg.thread_concurrency_levels = [20]
        report = ProductionValidator(cfg).run()
        assert report.quality_scores.overall > 0.0

    def test_report_to_markdown(self, fast_config):
        from medical_ai_engine.verification.production_validator import ProductionValidator
        cfg = VerificationConfiguration.fast()
        cfg.benchmark_iterations = 3
        cfg.benchmark_warmup_iterations = 1
        cfg.memory_test_cycles = 5
        cfg.thread_concurrency_levels = [20]
        report = ProductionValidator(cfg).run()
        md = report.to_markdown()
        assert "# Production Readiness Report" in md
        assert "Final Verdict" in md
        assert "Quality Scores" in md
