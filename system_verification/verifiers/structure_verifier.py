"""
system_verification/verifiers/structure_verifier.py
=====================================================
Verifies project directory hierarchy, required packages, configuration files,
entry points, documentation, and log files.

Author: Medical AI System — Module 5.5
"""

import logging
from typing import Any, Dict, List

import yaml

from system_verification.config import (
    EXPECTED_CONFIGS,
    EXPECTED_DOCS,
    EXPECTED_ENTRY_POINTS,
    EXPECTED_LOG_FILES,
    EXPECTED_PACKAGES,
    EXPECTED_ROOT_FILES,
    LOGS_DIR,
    PROJECT_ROOT,
)

logger = logging.getLogger(__name__)


class StructureVerifier:
    """Verifies project folder structure, configuration files, docs, and entry points."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run structure verification checks."""
        logger.info("Running StructureVerifier ...")
        checks: List[Dict[str, Any]] = []

        # 1. Packages
        for pkg in EXPECTED_PACKAGES:
            pkg_path = PROJECT_ROOT / pkg
            exists = pkg_path.exists() and pkg_path.is_dir()
            has_init = (pkg_path / "__init__.py").exists() if exists else False
            checks.append({
                "category": "Package",
                "item": pkg,
                "status": "PASS" if (exists and has_init) else ("WARNING" if exists else "FAIL"),
                "details": f"Directory exists: {exists}, __init__.py exists: {has_init}",
            })

        # 2. Configs
        for cfg in EXPECTED_CONFIGS:
            cfg_path = PROJECT_ROOT / cfg
            exists = cfg_path.exists()
            valid_yaml = False
            if exists:
                try:
                    with open(cfg_path, "r", encoding="utf-8") as fh:
                        yaml.safe_load(fh)
                    valid_yaml = True
                except Exception as exc:
                    logger.warning("Failed to parse YAML file %s: %s", cfg, exc)

            checks.append({
                "category": "Config File",
                "item": cfg,
                "status": "PASS" if valid_yaml else "FAIL",
                "details": f"File exists: {exists}, Valid YAML: {valid_yaml}",
            })

        # 3. Entry Points
        for ep in EXPECTED_ENTRY_POINTS:
            ep_path = PROJECT_ROOT / ep
            exists = ep_path.exists()
            checks.append({
                "category": "Entry Point",
                "item": ep,
                "status": "PASS" if exists else "FAIL",
                "details": f"File exists: {exists}",
            })

        # 4. Docs
        for doc in EXPECTED_DOCS:
            doc_path = PROJECT_ROOT / doc
            exists = doc_path.exists()
            checks.append({
                "category": "Documentation",
                "item": doc,
                "status": "PASS" if exists else "WARNING",
                "details": f"Doc file exists: {exists}",
            })

        # 5. Root Files
        for rf in EXPECTED_ROOT_FILES:
            rf_path = PROJECT_ROOT / rf
            exists = rf_path.exists()
            checks.append({
                "category": "Root File",
                "item": rf,
                "status": "PASS" if exists else "FAIL",
                "details": f"File exists: {exists}",
            })

        # 6. Log Files
        for lf in EXPECTED_LOG_FILES:
            lf_path = LOGS_DIR / lf
            exists = lf_path.exists()
            checks.append({
                "category": "Log File",
                "item": lf,
                "status": "PASS" if exists else "WARNING",
                "details": f"Log file exists: {exists}",
            })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "total": len(checks),
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info(
            "StructureVerifier complete: %d PASS, %d WARN, %d FAIL (Score: %.1f)",
            pass_count, warn_count, fail_count, self._results["score"]
        )
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
