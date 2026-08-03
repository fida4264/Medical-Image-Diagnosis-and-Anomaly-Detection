"""
test_benchmark.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Unit & integration tests for benchmark subsystem.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from medical_ai_engine.benchmark import (
    BenchmarkConfiguration,
    BenchmarkReport,
    BenchmarkRunner,
    ClassificationMetrics,
    ComparisonEngine,
    LeaderboardBuilder,
    ModalityBenchmarkResult,
    PerformanceMetrics,
    calculate_classification_metrics,
)
from medical_ai_engine.benchmark.benchmark_exceptions import (
    BenchmarkConfigurationError,
    MetricComputationError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def fast_cfg():
    return BenchmarkConfiguration.fast_test()


@pytest.fixture
def runner(fast_cfg):
    return BenchmarkRunner(config=fast_cfg)


@pytest.fixture
def all_results(runner):
    """Run full 7-modality benchmark once for use across multiple tests."""
    return runner.run_all_modalities()


# ── Configuration ─────────────────────────────────────────────────────────────

class TestBenchmarkConfiguration:
    def test_default_config(self):
        cfg = BenchmarkConfiguration.default()
        assert len(cfg.supported_modalities) == 7
        assert cfg.n_eval_runs == 50
        assert cfg.compute_roc_auc is True

    def test_fast_test_config(self):
        cfg = BenchmarkConfiguration.fast_test()
        assert cfg.n_eval_runs == 10
        assert cfg.n_warmup_runs == 2

    def test_invalid_n_eval_runs_raises(self):
        with pytest.raises(BenchmarkConfigurationError):
            BenchmarkConfiguration(n_eval_runs=0)

    def test_invalid_latency_budget_raises(self):
        with pytest.raises(BenchmarkConfigurationError):
            BenchmarkConfiguration(latency_budget_ms=0)

    def test_empty_modalities_raises(self):
        with pytest.raises(BenchmarkConfigurationError):
            BenchmarkConfiguration(supported_modalities=[])


# ── Metrics Math ──────────────────────────────────────────────────────────────

class TestClassificationMetrics:
    def test_perfect_binary(self):
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]
        m = calculate_classification_metrics(y_true, y_pred, class_names=["Neg", "Pos"])
        assert m.accuracy == pytest.approx(1.0)
        assert m.f1_score == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.recall == pytest.approx(1.0)

    def test_multiclass_metrics(self):
        rng = np.random.RandomState(42)
        n = 200
        y_true = rng.randint(0, 4, n)
        y_pred = y_true.copy()
        # Introduce 10% errors
        err_idx = rng.choice(n, size=20, replace=False)
        y_pred[err_idx] = (y_true[err_idx] + 1) % 4
        m = calculate_classification_metrics(
            y_true, y_pred, class_names=["A", "B", "C", "D"]
        )
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.f1_score <= 1.0
        assert len(m.class_names) == 4
        assert len(m.per_class_f1) == 4

    def test_roc_auc_computed(self):
        rng = np.random.RandomState(7)
        y_true = rng.randint(0, 2, 100)
        y_pred = y_true.copy()
        y_prob = np.zeros((100, 2))
        y_prob[np.arange(100), y_true] = 0.9
        y_prob[:, 1 - y_true] = 0.1
        m = calculate_classification_metrics(
            y_true, y_pred, y_prob=y_prob, class_names=["Neg", "Pos"]
        )
        assert m.roc_auc is not None
        assert m.roc_auc >= 0.5

    def test_zero_samples_raises(self):
        with pytest.raises(MetricComputationError):
            calculate_classification_metrics([], [])

    def test_to_dict_serializable(self):
        m = calculate_classification_metrics([0, 1, 0], [0, 1, 1])
        d = m.to_dict()
        assert "accuracy" in d and "f1_score" in d
        # Should be JSON serializable
        json.dumps(d)


# ── BenchmarkRunner ───────────────────────────────────────────────────────────

class TestBenchmarkRunner:
    def test_single_modality_returns_result(self, runner):
        res = runner.run_modality("chest_xray")
        assert res.modality == "chest_xray"
        assert 0.0 < res.classification_metrics.f1_score < 1.0
        assert 0.0 < res.classification_metrics.accuracy < 1.0
        assert res.performance_metrics.inference_latency_ms > 0

    def test_unknown_modality_raises(self, runner):
        from medical_ai_engine.benchmark.benchmark_exceptions import BenchmarkExecutionError
        with pytest.raises(BenchmarkExecutionError):
            runner.run_modality("invalid_modality")

    def test_all_7_modalities_run(self, all_results):
        assert len(all_results) == 7
        modalities = {r.modality for r in all_results}
        expected = {"brain_mri", "chest_xray", "brain_ct", "histopathology",
                    "retina", "skin", "ultrasound"}
        assert modalities == expected

    def test_every_result_has_roc_auc(self, all_results):
        for r in all_results:
            assert r.classification_metrics.roc_auc is not None

    def test_every_result_has_performance_metrics(self, all_results):
        for r in all_results:
            p = r.performance_metrics
            assert p.inference_latency_ms > 0
            assert p.throughput_rps > 0
            assert p.model_size_mb > 0
            assert p.param_count > 0

    def test_composite_scores_in_range(self, all_results):
        for r in all_results:
            assert 0.0 <= r.composite_score <= 1.0


# ── ComparisonEngine ──────────────────────────────────────────────────────────

class TestComparisonEngine:
    def test_comparison_produces_ranked_entries(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        assert summary.total_models_evaluated == 7
        ranks = [e.rank for e in summary.entries]
        assert ranks == list(range(1, 8))

    def test_top_entry_has_highest_composite(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        scores = [e.composite_score for e in summary.entries]
        assert scores[0] >= scores[-1]

    def test_empty_list_raises(self):
        from medical_ai_engine.benchmark.benchmark_exceptions import ComparisonEngineError
        engine = ComparisonEngine()
        with pytest.raises(ComparisonEngineError):
            engine.compare_results([])


# ── LeaderboardBuilder ────────────────────────────────────────────────────────

class TestLeaderboard:
    def test_leaderboard_builds(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        builder = LeaderboardBuilder()
        lb = builder.build(summary)
        assert len(lb.rows) == 7
        assert lb.overall_platform_grade in ("A+", "A", "B+", "B", "C", "D")

    def test_markdown_table_has_header(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        lb = LeaderboardBuilder().build(summary)
        md = lb.to_markdown_table()
        assert "| Rank | Tier |" in md

    def test_coverage_sums_to_n(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        lb = LeaderboardBuilder().build(summary)
        assert sum(lb.coverage.values()) == 7


# ── BenchmarkReport ───────────────────────────────────────────────────────────

class TestBenchmarkReport:
    def test_reports_are_generated(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        lb = LeaderboardBuilder().build(summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReport(output_directory=tmpdir)
            paths = reporter.generate(all_results, summary, lb, version="0.17.0.0")

            assert paths["markdown"].exists()
            assert paths["json"].exists()

            md_content = paths["markdown"].read_text(encoding="utf-8")
            assert "Medical AI Platform" in md_content
            assert "chest_xray" in md_content

            json_data = json.loads(paths["json"].read_text(encoding="utf-8"))
            assert json_data["version"] == "0.17.0.0"
            assert len(json_data["modality_results"]) == 7

    def test_markdown_contains_all_modalities(self, all_results):
        engine = ComparisonEngine()
        summary = engine.compare_results(all_results)
        lb = LeaderboardBuilder().build(summary)

        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = BenchmarkReport(output_directory=tmpdir)
            paths = reporter.generate(all_results, summary, lb)
            md = paths["markdown"].read_text(encoding="utf-8")

        for modality in ["brain_mri", "chest_xray", "brain_ct", "histopathology",
                          "retina", "skin", "ultrasound"]:
            assert modality in md, f"Modality {modality} missing from report"
