"""
tests/test_system_verification.py
===================================
Unit tests for Module 5.5: System Verification & Integration Testing.
"""

import pytest
import pandas as pd
from pathlib import Path

from system_verification.config import PROJECT_ROOT, VALIDATION_DIR
from system_verification.verifiers import (
    StructureVerifier,
    DatasetVerifier,
    MetadataVerifier,
    PreprocessingVerifier,
    AugmentationVerifier,
    EDAVerifier,
    IntegrationVerifier,
    PerformanceVerifier,
    CodeVerifier,
)
from system_verification.scoring import HealthScorer
from system_verification.reporters import ReportWriter, DashboardGenerator
from system_verification.runner import VerificationRunner


class TestVerifiers:
    """Test suite for all verifier components."""

    def test_structure_verifier(self):
        sv = StructureVerifier()
        res = sv.verify()
        assert "score" in res
        assert "checks" in res
        assert res["total"] > 0

    def test_dataset_verifier(self):
        dv = DatasetVerifier()
        res = dv.verify()
        assert "score" in res
        assert "modality_counts" in res

    def test_metadata_verifier(self):
        mv = MetadataVerifier()
        res = mv.verify()
        assert "score" in res
        assert "null_count" in res

    def test_preprocessing_verifier(self):
        pv = PreprocessingVerifier()
        res = pv.verify()
        assert "score" in res

    def test_augmentation_verifier(self):
        av = AugmentationVerifier()
        res = av.verify()
        assert "score" in res

    def test_eda_verifier(self):
        ev = EDAVerifier()
        res = ev.verify()
        assert "score" in res

    def test_integration_verifier(self):
        iv = IntegrationVerifier()
        res = iv.verify()
        assert "score" in res

    def test_performance_verifier(self):
        pv = PerformanceVerifier()
        res = pv.verify()
        assert "score" in res
        assert "cpu_cores" in res

    def test_code_verifier(self):
        cv = CodeVerifier()
        res = cv.verify()
        assert "score" in res


class TestScoringAndReporting:
    """Test suite for health scorer, report writer, dashboard generator, and runner."""

    def test_health_scorer(self):
        dummy_res = {
            "structure": {"score": 90},
            "dataset": {"score": 85},
            "metadata": {"score": 88},
            "preprocessing": {"score": 75},
            "augmentation": {"score": 70},
            "eda": {"score": 70},
            "integration": {"score": 90},
            "performance": {"score": 85},
            "code": {"imported_packages": 5, "test_files_count": 25},
        }
        scorer = HealthScorer(dummy_res)
        health_data = scorer.compute()
        assert "overall_score" in health_data
        assert 0 <= health_data["overall_score"] <= 100
        assert "decision" in health_data["readiness"]

    def test_report_writer(self, tmp_path):
        dummy_res = {"structure": {}, "dataset": {}, "metadata": {}}
        dummy_health = {
            "overall_score": 85.0,
            "scores": {"architecture": 90.0},
            "readiness": {"decision": "GO", "blockers": [], "recommendations": []},
        }
        writer = ReportWriter(dummy_res, dummy_health)
        saved = writer.write_all()
        assert len(saved) == 10
        for name, path in saved.items():
            assert path.exists()

    def test_dashboard_generator(self):
        dummy_res = {}
        dummy_health = {
            "overall_score": 82.5,
            "scores": {"architecture": 90.0},
            "readiness": {"decision": "CONDITIONAL GO", "blockers": ["Sample blocker"]},
        }
        dash_gen = DashboardGenerator(dummy_res, dummy_health)
        path = dash_gen.generate()
        assert path.exists()
        assert path.name == "validation_dashboard.html"

    def test_verification_runner(self):
        runner = VerificationRunner()
        res = runner.run()
        assert "verifier_results" in res
        assert "health_data" in res
        assert "saved_files" in res
        assert res["elapsed_seconds"] > 0
