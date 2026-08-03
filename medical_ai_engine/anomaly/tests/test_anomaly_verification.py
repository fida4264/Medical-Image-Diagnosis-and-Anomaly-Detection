"""
test_anomaly_verification.py — Unit & Integration tests for Module 16.7 Verification & Benchmarking.
"""
from __future__ import annotations

import pytest

from medical_ai_engine.anomaly.verification import (
    BenchmarkRunner,
    IntegrationValidator,
    MemoryProfiler,
    PerformanceProfiler,
    QualityValidator,
    StressTester,
    VerificationConfiguration,
    VerificationRunner,
)


class TestVerificationConfiguration:
    def test_default_config(self):
        cfg = VerificationConfiguration.default()
        assert cfg.latency_budget_ms == 50.0
        assert cfg.throughput_target_rps == 20.0

    def test_fast_test_config(self):
        cfg = VerificationConfiguration.fast_test()
        assert cfg.n_benchmark_iterations == 10
        assert cfg.stress_thread_counts == [5, 10]


class TestPerformanceProfiler:
    def test_profiler_run(self):
        cfg = VerificationConfiguration.fast_test()
        profiler = PerformanceProfiler(config=cfg)
        res = profiler.run()
        assert res.end_to_end_latency.count == 10
        assert res.throughput.total_requests == 10
        assert res.throughput.rps > 0


class TestMemoryProfiler:
    def test_memory_profiler_run(self):
        cfg = VerificationConfiguration.fast_test()
        profiler = MemoryProfiler(config=cfg)
        res = profiler.run()
        assert res.memory.baseline_mb >= 0
        assert res.within_budget is True


class TestStressTester:
    def test_stress_tester_tiers(self):
        cfg = VerificationConfiguration.fast_test()
        tester = StressTester(config=cfg)
        results = tester.run_all()
        assert len(results) == len(cfg.stress_thread_counts)
        for r in results:
            assert r.passed is True
            assert r.deadlock_detected is False
            assert r.race_condition_detected is False


class TestQualityValidator:
    def test_quality_validator_checks(self):
        cfg = VerificationConfiguration.fast_test()
        val = QualityValidator(config=cfg)
        res = val.run()
        assert res.overall_passed is True
        assert res.overall_score >= 0.80


class TestIntegrationValidator:
    def test_integration_validator_checks(self):
        cfg = VerificationConfiguration.fast_test()
        val = IntegrationValidator(config=cfg)
        res = val.run()
        assert res.overall_passed is True
        assert res.pass_count == len(res.checks)


class TestBenchmarkRunner:
    def test_benchmark_runner(self):
        cfg = VerificationConfiguration.fast_test()
        runner = BenchmarkRunner(config=cfg)
        res = runner.run()
        assert len(res.rows) >= 5
        table = res.as_markdown_table()
        assert "| Scenario | Mean (ms) |" in table


class TestVerificationRunnerFacade:
    def test_verification_runner_facade(self):
        cfg = VerificationConfiguration.fast_test()
        runner = VerificationRunner(config=cfg)
        res = runner.run_all_verifications()
        assert res["verdict"] in ("GO", "NO_GO")
        assert "report_path" in res
