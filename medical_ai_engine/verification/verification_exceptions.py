"""
Medical AI Engine — Verification Exceptions

Module 14.2.7: Production Verification & Stress Testing

Exception hierarchy for the verification subsystem.
Error codes: VER-700 to VER-705
"""

from __future__ import annotations
from typing import Any, Optional


class VerificationException(Exception):
    """
    Base exception for all verification failures.
    Error code: VER-700
    """
    error_code: str = "VER-700"

    def __init__(
        self,
        message: str,
        component: str = "Verification",
        cause: Optional[Exception] = None,
        details: Optional[Any] = None,
    ) -> None:
        self.component = component
        self.cause = cause
        self.details = details
        super().__init__(f"[{self.error_code}] [{component}] {message}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(code={self.error_code!r}, component={self.component!r})"


class ArchitectureAuditException(VerificationException):
    """
    Raised when the architecture audit detects critical violations.
    Error code: VER-701
    """
    error_code = "VER-701"

    def __init__(
        self,
        message: str,
        violations: Optional[list] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.violations = violations or []
        super().__init__(
            message=message,
            component="ArchitectureAuditor",
            cause=cause,
            details={"violations": self.violations},
        )


class StressTestException(VerificationException):
    """
    Raised when a stress test encounters a fatal infrastructure failure.
    Error code: VER-702
    """
    error_code = "VER-702"

    def __init__(
        self,
        message: str,
        scenario: str = "unknown",
        cause: Optional[Exception] = None,
    ) -> None:
        self.scenario = scenario
        super().__init__(
            message=message,
            component="StressTester",
            cause=cause,
            details={"scenario": scenario},
        )


class BenchmarkException(VerificationException):
    """
    Raised when a benchmark run fails to complete.
    Error code: VER-703
    """
    error_code = "VER-703"

    def __init__(
        self,
        message: str,
        benchmark_name: str = "unknown",
        cause: Optional[Exception] = None,
    ) -> None:
        self.benchmark_name = benchmark_name
        super().__init__(
            message=message,
            component="BenchmarkRunner",
            cause=cause,
            details={"benchmark": benchmark_name},
        )


class CompatibilityException(VerificationException):
    """
    Raised when a backward-compatibility violation is detected.
    Error code: VER-704
    """
    error_code = "VER-704"

    def __init__(
        self,
        message: str,
        breaking_changes: Optional[list] = None,
        cause: Optional[Exception] = None,
    ) -> None:
        self.breaking_changes = breaking_changes or []
        super().__init__(
            message=message,
            component="CompatibilityChecker",
            cause=cause,
            details={"breaking_changes": self.breaking_changes},
        )


class HealthCheckException(VerificationException):
    """
    Raised when the health monitor detects a critical system failure.
    Error code: VER-705
    """
    error_code = "VER-705"

    def __init__(
        self,
        message: str,
        health_score: float = 0.0,
        cause: Optional[Exception] = None,
    ) -> None:
        self.health_score = health_score
        super().__init__(
            message=message,
            component="HealthMonitor",
            cause=cause,
            details={"health_score": health_score},
        )
