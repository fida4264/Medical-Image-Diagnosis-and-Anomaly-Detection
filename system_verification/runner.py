"""
system_verification/runner.py
==============================
Master orchestrator for Module 5.5: System Verification & Integration Testing.

Pipeline Execution Steps:
[1/6] StructureVerifier — Directory skeleton, configs, entrypoints, docs, logs
[2/6] DatasetVerifier   — Dataset integrity, modalities, corruption, mask files
[3/6] MetadataVerifier  — master_metadata.csv schema, nulls, duplicates, Gini
[4/6] Module Verifiers  — Preprocessing, Augmentation, EDA verifiers
[5/6] Integration & Perf— IntegrationVerifier, PerformanceVerifier, CodeVerifier
[6/6] Scoring & Reports — HealthScorer, ReportWriter, DashboardGenerator

Author: Medical AI System — Module 5.5
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from system_verification.config import LOG_FILE_NAME, LOGS_DIR
from system_verification.reporters import DashboardGenerator, ReportWriter
from system_verification.scoring import HealthScorer
from system_verification.verifiers import (
    AugmentationVerifier,
    CodeVerifier,
    DatasetVerifier,
    EDAVerifier,
    IntegrationVerifier,
    MetadataVerifier,
    PerformanceVerifier,
    PreprocessingVerifier,
    StructureVerifier,
)

# ---------------------------------------------------------------------------
# Logging Setup
# ---------------------------------------------------------------------------
LOGS_DIR.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)-8s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(LOGS_DIR / LOG_FILE_NAME, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def _step(n: int, total: int, msg: str) -> None:
    logger.info("[%d/%d] %s", n, total, msg)
    print(f"\n{'=' * 78}\n  [{n}/{total}] {msg}\n{'=' * 78}")


class VerificationRunner:
    """Orchestrates end-to-end system verification, health scoring, and report generation."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def run(self) -> Dict[str, Any]:
        """Execute all verifiers and produce reports."""
        t0 = time.perf_counter()
        print("\n" + "=" * 78)
        print("   MODULE 5.5: SYSTEM VERIFICATION & INTEGRATION TESTING")
        print("=" * 78)

        # 1. Structure Verifier
        _step(1, 6, "Verifying Project Structure & Hierarchy")
        sv = StructureVerifier()
        self._results["structure"] = sv.verify()

        # 2. Dataset Verifier
        _step(2, 6, "Verifying Raw Dataset Integrity & Modalities")
        dv = DatasetVerifier()
        self._results["dataset"] = dv.verify()

        # 3. Metadata Verifier
        _step(3, 6, "Verifying Master Metadata Schema & Balance")
        mv = MetadataVerifier()
        self._results["metadata"] = mv.verify()

        # 4. Preprocessing, Augmentation & EDA Verifiers
        _step(4, 6, "Verifying Preprocessing, Augmentation & EDA Outputs")
        pv = PreprocessingVerifier()
        self._results["preprocessing"] = pv.verify()

        av = AugmentationVerifier()
        self._results["augmentation"] = av.verify()

        ev = EDAVerifier()
        self._results["eda"] = ev.verify()

        # 5. Integration, Performance & Code Verifiers
        _step(5, 6, "Verifying Module Integration, Performance & Code Quality")
        iv = IntegrationVerifier()
        self._results["integration"] = iv.verify()

        perf_v = PerformanceVerifier()
        self._results["performance"] = perf_v.verify()

        cv = CodeVerifier()
        self._results["code"] = cv.verify()

        # 6. Health Scorer & Report Generation
        _step(6, 6, "Computing Health Score & Generating Validation Reports")
        hs = HealthScorer(self._results)
        health_data = hs.compute()
        self._results["health"] = health_data

        writer = ReportWriter(self._results, health_data)
        saved_files = writer.write_all()

        dashboard_gen = DashboardGenerator(self._results, health_data)
        dashboard_path = dashboard_gen.generate()
        saved_files["dashboard"] = dashboard_path

        t1 = time.perf_counter()
        elapsed = round(t1 - t0, 2)

        overall_score = health_data.get("overall_score", 0.0)
        decision = health_data.get("readiness", {}).get("decision", "N/A")

        print("\n" + "=" * 78)
        print(f"  SYSTEM VERIFICATION COMPLETE in {elapsed}s")
        print(f"  OVERALL PROJECT HEALTH SCORE: {overall_score:.1f} / 100")
        print(f"  MODULE 6 READINESS DECISION:  {decision}")
        print("=" * 78)
        print(f"  Reports written to: {dashboard_path.parent}")
        print("=" * 78 + "\n")

        return {
            "verifier_results": self._results,
            "health_data": health_data,
            "saved_files": saved_files,
            "elapsed_seconds": elapsed,
        }
