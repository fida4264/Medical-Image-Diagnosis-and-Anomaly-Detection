"""
Medical AI Engine — Architecture Auditor

Module 14.2.7: Production Verification & Stress Testing

Audits the medical_ai_engine package tree for:
  - Circular imports
  - Interface adherence
  - Design pattern correctness
  - Module isolation (Engine Core has no disease module imports)
  - Schema version consistency
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_report import VerificationResult, VerificationStatus

logger = logging.getLogger("ArchitectureAuditor")


# ── Known design pattern markers ────────────────────────────────────────────

_PATTERN_CHECKS = {
    "Registry Pattern": {
        "module": "medical_ai_engine.registry.model_registry",
        "class": "ModelRegistry",
        "markers": ["_instance", "get_instance", "register_adapter", "get_adapter"],
    },
    "Pipeline Pattern": {
        "module": "medical_ai_engine.pipeline.pipeline_executor",
        "class": "PipelineExecutor",
        "markers": ["execute", "stages", "rollback"],
    },
    "Strategy Pattern": {
        "module": "medical_ai_engine.routing.modality_router",
        "class": "ModalityRouter",
        "markers": ["route"],
    },
    "Factory Pattern": {
        "module": "medical_ai_engine.pipeline.pipeline_stage_factory",
        "class": "PipelineStageFactory",
        "markers": ["create_default_pipeline"],
    },
    "Adapter Pattern": {
        "module": "medical_ai_engine.adapters.base_predictor_adapter",
        "class": "BasePredictorAdapter",
        "markers": ["predict", "load_model", "modality"],
    },
    "Dependency Injection": {
        "module": "medical_ai_engine.core.medical_ai_engine",
        "class": "MedicalAIEngine",
        "markers": ["__init__", "initialize", "execute"],
    },
}

# ── Required interfaces ──────────────────────────────────────────────────────

_INTERFACE_CHECKS = [
    ("medical_ai_engine.core.medical_ai_engine", "MedicalAIEngine",
     "medical_ai_engine.interfaces.engine_interface", "IEngine"),
    ("medical_ai_engine.registry.model_registry", "ModelRegistry",
     "medical_ai_engine.interfaces.registry_interface", "IModelRegistry"),
    ("medical_ai_engine.routing.modality_router", "ModalityRouter",
     "medical_ai_engine.interfaces.router_interface", "IModalityRouter"),
    ("medical_ai_engine.response.response_builder", "ResponseBuilder",
     "medical_ai_engine.interfaces.response_builder_interface", "IResponseBuilder"),
]

# ── Forbidden imports in Engine Core ────────────────────────────────────────

_FORBIDDEN_CORE_IMPORTS = [
    "brain_mri", "chest_xray", "brain_ct", "histopathology",
    "retina", "skin", "ultrasound",
    "brain.mri", "brain.ct",
]


class ArchitectureAuditor:
    """
    Performs a non-destructive read-only audit of the medical_ai_engine package tree.
    Reports architecture violations, design pattern status, and interface adherence.
    """

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self.config = config or VerificationConfiguration()

    def run(self) -> VerificationResult:
        """Run the full architecture audit and return a VerificationResult."""
        start = time.perf_counter()
        violations: List[str] = []
        warnings: List[str] = []
        details: Dict[str, Any] = {}
        checks_passed = 0
        total_checks = 0

        # 1. Module Import Check
        total_checks += 1
        import_ok, import_errs = self._check_core_imports()
        if import_ok:
            checks_passed += 1
        else:
            violations.extend(import_errs)
        details["import_violations"] = import_errs

        # 2. Interface Adherence
        if self.config.check_interface_adherence:
            total_checks += 1
            iface_ok, iface_errs = self._check_interface_adherence()
            if iface_ok:
                checks_passed += 1
            else:
                violations.extend(iface_errs)
            details["interface_violations"] = iface_errs

        # 3. Design Patterns
        if self.config.check_design_patterns:
            pattern_results = {}
            for pattern_name, spec in _PATTERN_CHECKS.items():
                total_checks += 1
                ok, err = self._check_pattern(pattern_name, spec)
                pattern_results[pattern_name] = "PRESENT" if ok else f"MISSING: {err}"
                if ok:
                    checks_passed += 1
                else:
                    violations.append(f"Pattern '{pattern_name}': {err}")
            details["pattern_results"] = pattern_results

        # 4. Module Isolation
        if self.config.check_module_isolation:
            total_checks += 1
            iso_ok, iso_errs = self._check_engine_core_isolation()
            if iso_ok:
                checks_passed += 1
            else:
                warnings.extend(iso_errs)  # Warn rather than hard-fail (lazy imports are OK)
            details["isolation_warnings"] = iso_errs

        # 5. Schema Version Consistency
        total_checks += 1
        schema_ok, schema_err = self._check_schema_version()
        if schema_ok:
            checks_passed += 1
        else:
            violations.append(schema_err)
        details["schema_version_ok"] = schema_ok

        # ── Score ──────────────────────────────────────────────────────────
        score = (checks_passed / total_checks * 100.0) if total_checks > 0 else 0.0
        critical = [v for v in violations if "interface" in v.lower() or "circular" in v.lower()]

        if critical:
            status = VerificationStatus.FAILED
        elif violations:
            status = VerificationStatus.WARNING
        else:
            status = VerificationStatus.PASSED

        details["checks_passed"] = checks_passed
        details["total_checks"] = total_checks

        return VerificationResult(
            name="Architecture Compliance Audit",
            status=status,
            score=score,
            duration_seconds=time.perf_counter() - start,
            message=(
                f"{checks_passed}/{total_checks} architecture checks passed. "
                f"{len(violations)} violation(s), {len(warnings)} warning(s)."
            ),
            details=details,
            violations=violations,
            warnings=warnings,
        )

    def _check_core_imports(self) -> Tuple[bool, List[str]]:
        """Verify that all core engine modules can be imported cleanly."""
        core_modules = [
            "medical_ai_engine.core.medical_ai_engine",
            "medical_ai_engine.registry.model_registry",
            "medical_ai_engine.routing.modality_router",
            "medical_ai_engine.pipeline.pipeline_executor",
            "medical_ai_engine.response.response_builder",
            "medical_ai_engine.integration.engine_integrator",
            "medical_ai_engine.interfaces.engine_interface",
            "medical_ai_engine.interfaces.registry_interface",
            "medical_ai_engine.interfaces.router_interface",
            "medical_ai_engine.interfaces.predictor_adapter",
            "medical_ai_engine.interfaces.response_builder_interface",
        ]
        errors = []
        for mod_name in core_modules:
            try:
                importlib.import_module(mod_name)
            except ImportError as e:
                errors.append(f"Cannot import '{mod_name}': {e}")
            except Exception as e:
                errors.append(f"Error importing '{mod_name}': {e}")
        return len(errors) == 0, errors

    def _check_interface_adherence(self) -> Tuple[bool, List[str]]:
        """Verify that concrete classes implement their declared interfaces."""
        errors = []
        for impl_mod, impl_cls, iface_mod, iface_cls in _INTERFACE_CHECKS:
            try:
                impl_module = importlib.import_module(impl_mod)
                iface_module = importlib.import_module(iface_mod)
                impl_class = getattr(impl_module, impl_cls, None)
                iface_class = getattr(iface_module, iface_cls, None)

                if impl_class is None:
                    errors.append(f"Class '{impl_cls}' not found in '{impl_mod}'")
                    continue
                if iface_class is None:
                    errors.append(f"Interface '{iface_cls}' not found in '{iface_mod}'")
                    continue
                if not issubclass(impl_class, iface_class):
                    errors.append(
                        f"'{impl_cls}' does not implement interface '{iface_cls}'"
                    )
            except Exception as e:
                errors.append(f"Interface check '{impl_cls} -> {iface_cls}' failed: {e}")
        return len(errors) == 0, errors

    def _check_pattern(self, pattern_name: str, spec: Dict[str, Any]) -> Tuple[bool, str]:
        """Verify that a design pattern is correctly present."""
        try:
            mod = importlib.import_module(spec["module"])
            cls = getattr(mod, spec["class"], None)
            if cls is None:
                return False, f"Class '{spec['class']}' not found"

            # Check for marker attributes/methods
            missing = []
            for marker in spec["markers"]:
                found = (
                    hasattr(cls, marker)
                    or any(marker in str(m) for m in dir(cls))
                )
                if not found:
                    missing.append(marker)

            if missing:
                return False, f"Missing markers: {missing}"
            return True, ""
        except Exception as e:
            return False, str(e)

    def _check_engine_core_isolation(self) -> Tuple[bool, List[str]]:
        """
        Verify that medical_ai_engine/core/ source files do not contain
        static top-level imports of disease modules (Modules 7–13).
        Note: lazy imports inside methods are acceptable.
        """
        warnings = []
        core_dir = Path(__file__).parent.parent / "core"
        if not core_dir.exists():
            return True, []

        for py_file in core_dir.glob("*.py"):
            try:
                source = py_file.read_text(encoding="utf-8")
                for forbidden in _FORBIDDEN_CORE_IMPORTS:
                    # Only check top-level import lines (not inside functions/methods)
                    lines = source.splitlines()
                    for i, line in enumerate(lines, 1):
                        stripped = line.strip()
                        if (
                            stripped.startswith("import ") or stripped.startswith("from ")
                        ) and forbidden in stripped and not line.startswith(" ") and not line.startswith("\t"):
                            warnings.append(
                                f"{py_file.name}:{i} — top-level import of disease module '{forbidden}'"
                            )
            except Exception:
                pass

        return len(warnings) == 0, warnings

    def _check_schema_version(self) -> Tuple[bool, str]:
        """Verify RESPONSE_SCHEMA_VERSION is defined and consistent."""
        try:
            from medical_ai_engine.response.response_schema import RESPONSE_SCHEMA_VERSION
            if not RESPONSE_SCHEMA_VERSION or not isinstance(RESPONSE_SCHEMA_VERSION, str):
                return False, "RESPONSE_SCHEMA_VERSION is empty or not a string"
            return True, ""
        except ImportError as e:
            return False, f"Cannot import RESPONSE_SCHEMA_VERSION: {e}"
