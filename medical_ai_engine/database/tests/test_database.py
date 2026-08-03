"""
test_database.py — Module 19: Prediction History & Database
Medical AI Platform — Unit & integration test suite for database subsystem.
"""

from __future__ import annotations

import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from medical_ai_engine.database import (
    DatabaseConfiguration,
    DatabaseManager,
    HistoryManager,
    PlatformDatabaseStatistics,
    PredictionRecord,
    PredictionRepository,
    QueryEngine,
    StatisticsEngine,
)
from medical_ai_engine.database.database_exceptions import (
    DatabaseConfigurationError,
    RepositoryError,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def memory_cfg():
    return DatabaseConfiguration.fast_test()


@pytest.fixture
def db_manager(memory_cfg):
    mgr = DatabaseManager(memory_cfg)
    yield mgr
    mgr.close()


@pytest.fixture
def repo(db_manager):
    return PredictionRepository(db_manager)


@pytest.fixture
def history_mgr(db_manager):
    return HistoryManager(db_manager=db_manager)


# ── Test Database Configuration & Manager ────────────────────────────────────

class TestDatabaseConfigurationAndManager:
    def test_default_config(self):
        cfg = DatabaseConfiguration.default()
        assert cfg.database_path == "medical_ai.db"
        assert cfg.auto_create_tables is True

    def test_invalid_timeout_raises(self):
        with pytest.raises(DatabaseConfigurationError):
            DatabaseConfiguration(connection_timeout_seconds=0)

    def test_schema_initialization(self, db_manager):
        with db_manager.connection_scope() as conn:
            cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='predictions';")
            assert cursor.fetchone() is not None


# ── Test Prediction Repository (CRUD) ────────────────────────────────────────

class TestPredictionRepository:
    def test_save_and_get_by_id(self, repo):
        rec = PredictionRecord(
            prediction_id="REQ-001",
            patient_id="PT-100",
            modality="brain_mri",
            predicted_disease="Glioma",
            confidence=0.965,
        )
        saved_id = repo.save(rec)
        assert saved_id == "REQ-001"

        fetched = repo.get_by_id("REQ-001")
        assert fetched is not None
        assert fetched.predicted_disease == "Glioma"
        assert fetched.confidence == pytest.approx(0.965)
        assert fetched.modality == "brain_mri"

    def test_update_record(self, repo):
        rec = PredictionRecord(
            prediction_id="REQ-002",
            predicted_disease="Normal",
            confidence=0.90,
        )
        repo.save(rec)

        rec_updated = PredictionRecord(
            prediction_id="REQ-002",
            predicted_disease="Pneumonia",
            confidence=0.97,
        )
        repo.save(rec_updated)

        fetched = repo.get_by_id("REQ-002")
        assert fetched.predicted_disease == "Pneumonia"
        assert fetched.confidence == pytest.approx(0.97)

    def test_delete_record(self, repo):
        rec = PredictionRecord(prediction_id="REQ-003")
        repo.save(rec)
        assert repo.get_by_id("REQ-003") is not None

        assert repo.delete("REQ-003") is True
        assert repo.get_by_id("REQ-003") is None

    def test_count_records(self, repo):
        assert repo.count() == 0
        repo.save(PredictionRecord(prediction_id="C1"))
        repo.save(PredictionRecord(prediction_id="C2"))
        assert repo.count() == 2


# ── Test Query Engine & Filters ──────────────────────────────────────────────

class TestQueryEngine:
    def test_search_filtering(self, db_manager):
        repo = PredictionRepository(db_manager)
        q_engine = QueryEngine(db_manager)

        # Seed records
        repo.save(PredictionRecord(prediction_id="P1", modality="chest_xray", predicted_disease="Pneumonia", risk_level="LOW", confidence=0.95))
        repo.save(PredictionRecord(prediction_id="P2", modality="brain_mri", predicted_disease="Glioma", risk_level="HIGH", confidence=0.88))
        repo.save(PredictionRecord(prediction_id="P3", modality="chest_xray", predicted_disease="COVID-19", risk_level="CRITICAL", confidence=0.92))

        # Filter by modality
        cxr_recs = q_engine.search(modality="chest_xray")
        assert len(cxr_recs) == 2
        assert {r.prediction_id for r in cxr_recs} == {"P1", "P3"}

        # Filter by risk level
        critical_recs = q_engine.search(risk_level="CRITICAL")
        assert len(critical_recs) == 1
        assert critical_recs[0].prediction_id == "P3"

        # Filter by min confidence
        high_conf = q_engine.search(min_confidence=0.90)
        assert len(high_conf) == 2

    def test_get_latest(self, db_manager):
        repo = PredictionRepository(db_manager)
        q_engine = QueryEngine(db_manager)

        for i in range(15):
            repo.save(PredictionRecord(prediction_id=f"L-{i}", created_at=f"2026-08-02T12:00:{i:02d}"))

        latest = q_engine.get_latest(limit=5)
        assert len(latest) == 5
        assert latest[0].prediction_id == "L-14"


# ── Test Statistics Engine ────────────────────────────────────────────────────

class TestStatisticsEngine:
    def test_compute_statistics(self, db_manager):
        repo = PredictionRepository(db_manager)
        stat_engine = StatisticsEngine(db_manager)

        repo.save(PredictionRecord(prediction_id="S1", modality="chest_xray", predicted_disease="Pneumonia", confidence=0.90, processing_time_ms=10.0, risk_level="LOW"))
        repo.save(PredictionRecord(prediction_id="S2", modality="chest_xray", predicted_disease="COVID-19", confidence=0.94, processing_time_ms=20.0, risk_level="HIGH"))
        repo.save(PredictionRecord(prediction_id="S3", modality="brain_mri", predicted_disease="Glioma", confidence=0.86, processing_time_ms=30.0, risk_level="HIGH"))

        stats = stat_engine.compute_statistics()
        assert isinstance(stats, PlatformDatabaseStatistics)
        assert stats.total_predictions == 3
        assert stats.predictions_per_modality["chest_xray"] == 2
        assert stats.predictions_per_modality["brain_mri"] == 1
        assert stats.average_confidence == pytest.approx(0.90, abs=0.01)
        assert stats.average_processing_time_ms == pytest.approx(20.0, abs=0.1)
        assert stats.risk_level_distribution["HIGH"] == 2


# ── Test History Manager & Exports ───────────────────────────────────────────

class TestHistoryManagerAndExports:
    def test_record_prediction_cycle(self, history_mgr):
        dict_pred = {
            "prediction_id": "CYCLE-01",
            "patient_id": "PT-77",
            "modality": "retina",
            "predicted_disease": "Mild DR",
            "confidence": 0.89,
            "processing_time_ms": 15.0,
        }
        exp = {"summary": "Retinal fovea focus", "heatmap_path": "art/retina.png"}
        anom = {"status": "KNOWN_DISEASE", "risk_level": "LOW", "risk_score": 0.08}

        rec_id = history_mgr.record_prediction(
            prediction_result=dict_pred,
            explainability_result=exp,
            anomaly_result=anom,
            report_path="docs/rpt.pdf",
        )

        assert rec_id == "CYCLE-01"

        fetched = history_mgr.get_prediction("CYCLE-01")
        assert fetched is not None
        assert fetched.predicted_disease == "Mild DR"
        assert fetched.report_path == "docs/rpt.pdf"

    def test_csv_and_json_exports(self, history_mgr):
        for i in range(5):
            history_mgr.repository.save(PredictionRecord(
                prediction_id=f"EXP-{i}",
                modality="skin",
                predicted_disease="Melanoma" if i % 2 == 0 else "Nevus",
            ))

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_p = Path(tmpdir) / "test_history.csv"
            json_p = Path(tmpdir) / "test_history.json"

            history_mgr.export_csv(csv_p)
            history_mgr.export_json(json_p)

            assert csv_p.exists()
            assert json_p.exists()

            # Verify CSV content
            lines = csv_p.read_text(encoding="utf-8").splitlines()
            assert len(lines) == 6  # header + 5 records
            assert "prediction_id,created_at" in lines[0]

            # Verify JSON content
            data = json.loads(json_p.read_text(encoding="utf-8"))
            assert len(data) == 5
            assert data[0]["modality"] == "skin"


# ── Test Thread Safety ────────────────────────────────────────────────────────

class TestThreadSafety:
    def test_concurrent_inserts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "concurrent.db")
            cfg = DatabaseConfiguration(database_path=db_path, enable_wal_mode=True)
            mgr = DatabaseManager(cfg)
            h_mgr = HistoryManager(db_manager=mgr)

            def _insert_worker(worker_id: int):
                try:
                    for i in range(10):
                        p_id = f"W{worker_id}-P{i}"
                        h_mgr.repository.save(PredictionRecord(
                            prediction_id=p_id,
                            patient_id=f"PT-{worker_id}",
                            modality="chest_xray",
                            predicted_disease="Pneumonia",
                        ))
                finally:
                    mgr.close()

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(_insert_worker, wid) for wid in range(5)]
                for f in futures:
                    f.result()

            assert h_mgr.repository.count() == 50
            mgr.close()
