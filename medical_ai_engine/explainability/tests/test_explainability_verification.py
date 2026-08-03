"""
test_explainability_verification.py — Module 15.7: Explainability Verification & Benchmarking
Medical AI Platform — Unit test suite for verification, benchmarking, stress testing, and readiness.
"""

from __future__ import annotations

import pytest

from medical_ai_engine.explainability.verification import (
    ArchitectureValidator,
    BenchmarkRunner,
    CategoryScore,
    ConsistencyChecker,
    FaithfulnessEvaluator,
    MemoryProfiler,
    PerformanceProfiler,
    ProductionValidator,
    QualityEvaluator,
    StressTester,
    SystemIntegrationValidator,
    VerificationConfiguration,
    VerificationMetrics,
    VerificationReportGenerator,
    VerificationRunner,
)


@pytest.fixture
def fast_config() -> VerificationConfiguration:
    return VerificationConfiguration.fast_test()


class TestVerificationComponents:

    def test_architecture_validator(self):
        val = ArchitectureValidator()
        score = CategoryScore(name="arch")
        val.validate(score)
        assert score.passed > 0

    def test_system_integration_validator(self):
        val = SystemIntegrationValidator()
        score = CategoryScore(name="integ")
        val.validate(score)
        assert score.passed > 0

    def test_quality_evaluator(self):
        evaluator = QualityEvaluator()
        score = CategoryScore(name="quality")
        evaluator.evaluate(score)
        assert score.passed > 0

    def test_faithfulness_evaluator(self):
        evaluator = FaithfulnessEvaluator()
        score = CategoryScore(name="faith")
        evaluator.evaluate(score)
        assert score.passed > 0

    def test_consistency_checker(self):
        checker = ConsistencyChecker()
        score = CategoryScore(name="consistency")
        checker.check(score)
        assert score.passed > 0

    def test_performance_profiler(self):
        profiler = PerformanceProfiler()
        score = CategoryScore(name="perf")
        metrics = VerificationMetrics()
        profiler.profile(score, metrics)
        assert score.passed > 0
        assert len(metrics.latency_samples_ms) > 0

    def test_benchmark_runner(self):
        runner = BenchmarkRunner()
        score = CategoryScore(name="bench")
        res = runner.run(score)
        assert "brain_mri" in res
        assert res["brain_mri"]["rps"] > 0

    def test_memory_profiler(self):
        profiler = MemoryProfiler()
        score = CategoryScore(name="mem")
        metrics = VerificationMetrics()
        profiler.profile(score, metrics)
        assert score.passed > 0

    def test_stress_tester(self):
        tester = StressTester()
        score = CategoryScore(name="stress")
        tester.test(score, concurrency_levels=[5, 10])
        assert score.passed >= 2

    def test_production_validator(self):
        val = ProductionValidator()
        metrics = VerificationMetrics()
        cat = metrics.get_category("architecture")
        cat.record_pass("test pass")
        res = val.evaluate_verdict(metrics)
        assert res["verdict"] in ("GO", "NO GO")

    def test_verification_runner_full_run(self, fast_config):
        runner = VerificationRunner(config=fast_config)
        res = runner.run_all()
        assert res["verdict"] == "GO"
        assert res["overall_score_pct"] >= 90.0
