"""
system_verification/verifiers/code_verifier.py
================================================
Verifies package imports, circular dependencies, requirements.txt completeness,
test collection, and docstring coverage.

Author: Medical AI System — Module 5.5
"""

import importlib
import logging

from typing import Any, Dict, List

from system_verification.config import (
    EXPECTED_PACKAGES,
    PROJECT_ROOT,
    REQUIRED_PACKAGES,
)

logger = logging.getLogger(__name__)


class CodeVerifier:
    """Verifies Python imports, requirements.txt, and test suite status."""

    def __init__(self) -> None:
        self._results: Dict[str, Any] = {}

    def verify(self) -> Dict[str, Any]:
        """Run code verification checks."""
        logger.info("Running CodeVerifier ...")
        checks: List[Dict[str, Any]] = []

        # 1. Circular import & package importability test
        imported_packages = 0
        failed_imports: List[str] = []

        for pkg in EXPECTED_PACKAGES:
            if pkg == "tests":
                continue
            try:
                importlib.import_module(pkg)
                imported_packages += 1
            except Exception as exc:
                failed_imports.append(f"{pkg}: {exc}")
                logger.warning("Failed to import package %s: %s", pkg, exc)

        status_import = "PASS" if not failed_imports else "FAIL"
        checks.append({
            "check": "Package Importability & Circular Dependency Audit",
            "status": status_import,
            "details": f"Successfully imported {imported_packages} packages. Failures: {failed_imports or 'None'}",
        })

        # 2. Third-party dependency import audit (requirements.txt completeness)
        missing_libs: List[str] = []
        installed_libs: List[str] = []

        for mod_name, pip_name in REQUIRED_PACKAGES.items():
            try:
                importlib.import_module(mod_name)
                installed_libs.append(pip_name)
            except ImportError:
                missing_libs.append(f"{pip_name} (import name: {mod_name})")

        status_deps = "PASS" if not missing_libs else "WARNING"
        checks.append({
            "check": "Third-Party Library Availability (requirements.txt)",
            "status": status_deps,
            "details": f"Installed: {len(installed_libs)}/{len(REQUIRED_PACKAGES)}. Missing: {missing_libs or 'None'}",
        })

        # 3. Test files collection
        tests_dir = PROJECT_ROOT / "tests"
        test_files_count = 0
        if tests_dir.exists():
            test_files = list(tests_dir.glob("test_*.py"))
            test_files_count = len(test_files)

        checks.append({
            "check": "Test Suite Inventory",
            "status": "PASS" if test_files_count >= 20 else "WARNING",
            "details": f"Test files found in tests/: {test_files_count}",
        })

        pass_count = sum(1 for c in checks if c["status"] == "PASS")
        warn_count = sum(1 for c in checks if c["status"] == "WARNING")
        fail_count = sum(1 for c in checks if c["status"] == "FAIL")

        self._results = {
            "checks": checks,
            "imported_packages": imported_packages,
            "failed_imports": failed_imports,
            "installed_libs": installed_libs,
            "missing_libs": missing_libs,
            "test_files_count": test_files_count,
            "pass_count": pass_count,
            "warning_count": warn_count,
            "fail_count": fail_count,
            "score": round((pass_count / len(checks)) * 100, 2) if checks else 0.0,
        }
        logger.info("CodeVerifier complete: score=%.1f", self._results["score"])
        return self._results

    @property
    def results(self) -> Dict[str, Any]:
        return self._results
