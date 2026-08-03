"""
Medical AI Engine — Verification Report Data Classes

Module 14.2.7: Production Verification & Stress Testing

Data classes for all verification results, quality scores, and the
final production readiness report with GO / NO GO decision logic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class VerificationStatus(str, Enum):
    """Status of an individual verification check."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class ProductionVerdict(str, Enum):
    """Final GO / NO GO verdict for production readiness."""
    GO = "GO"
    NO_GO = "NO GO"
    CONDITIONAL_GO = "CONDITIONAL GO"


@dataclass
class VerificationResult:
    """Result of one individual verification check."""
    name: str
    status: VerificationStatus
    score: float = 0.0           # 0–100
    duration_seconds: float = 0.0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == VerificationStatus.PASSED

    @property
    def failed(self) -> bool:
        return self.status == VerificationStatus.FAILED

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def summary_line(self) -> str:
        icon = {"PASSED": "✔", "FAILED": "✘", "WARNING": "⚠", "SKIPPED": "—", "ERROR": "!"}.get(
            self.status.value, "?"
        )
        return f"  {icon}  {self.name:<40}  {self.status.value:<10}  score={self.score:.1f}/100  ({self.duration_seconds:.2f}s)"


@dataclass
class QualityScores:
    """10-dimension quality scoring of the platform."""
    architecture: float = 0.0
    engineering: float = 0.0
    maintainability: float = 0.0
    scalability: float = 0.0
    performance: float = 0.0
    reliability: float = 0.0
    security_readiness: float = 0.0
    documentation: float = 0.0
    testing: float = 0.0
    overall: float = 0.0

    def compute_overall(self) -> float:
        """Compute weighted overall score."""
        weights = {
            "architecture": 0.15,
            "engineering": 0.15,
            "maintainability": 0.10,
            "scalability": 0.10,
            "performance": 0.15,
            "reliability": 0.15,
            "security_readiness": 0.05,
            "documentation": 0.05,
            "testing": 0.10,
        }
        total = (
            self.architecture * weights["architecture"]
            + self.engineering * weights["engineering"]
            + self.maintainability * weights["maintainability"]
            + self.scalability * weights["scalability"]
            + self.performance * weights["performance"]
            + self.reliability * weights["reliability"]
            + self.security_readiness * weights["security_readiness"]
            + self.documentation * weights["documentation"]
            + self.testing * weights["testing"]
        )
        self.overall = round(total, 2)
        return self.overall

    def grade(self) -> str:
        """Letter grade for overall score."""
        s = self.overall
        if s >= 90:
            return "A+"
        elif s >= 85:
            return "A"
        elif s >= 80:
            return "A-"
        elif s >= 75:
            return "B+"
        elif s >= 70:
            return "B"
        elif s >= 65:
            return "B-"
        elif s >= 60:
            return "C+"
        elif s >= 50:
            return "C"
        else:
            return "F"

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)

    def to_table(self) -> str:
        rows = [
            ("Architecture", self.architecture),
            ("Engineering", self.engineering),
            ("Maintainability", self.maintainability),
            ("Scalability", self.scalability),
            ("Performance", self.performance),
            ("Reliability", self.reliability),
            ("Security Readiness", self.security_readiness),
            ("Documentation", self.documentation),
            ("Testing", self.testing),
            ("OVERALL", self.overall),
        ]
        header = f"{'Dimension':<25} {'Score':>8} {'Grade':>6}"
        sep = "-" * 45
        lines = [header, sep]
        for name, score in rows:
            grade = self.grade() if name == "OVERALL" else _letter_grade(score)
            lines.append(f"{name:<25} {score:>7.1f}% {grade:>6}")
        return "\n".join(lines)


def _letter_grade(score: float) -> str:
    if score >= 90:
        return "A+"
    elif score >= 85:
        return "A"
    elif score >= 80:
        return "A-"
    elif score >= 75:
        return "B+"
    elif score >= 70:
        return "B"
    elif score >= 65:
        return "B-"
    elif score >= 60:
        return "C+"
    elif score >= 50:
        return "C"
    else:
        return "F"


@dataclass
class PerformanceSummary:
    """Aggregated performance metrics from benchmark runs."""
    avg_latency_ms: float = 0.0
    min_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    p50_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    throughput_rps: float = 0.0
    peak_memory_mb: float = 0.0
    avg_cpu_pct: float = 0.0
    per_modality: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class StressSummary:
    """Aggregated results from stress tests."""
    burst_pass: bool = False
    sustained_pass: bool = False
    ramp_pass: bool = False
    error_injection_pass: bool = False
    registry_pressure_pass: bool = False
    max_concurrency_achieved: int = 0
    total_requests: int = 0
    total_failures: int = 0
    failure_rate: float = 0.0
    max_latency_under_load_ms: float = 0.0

    @property
    def all_passed(self) -> bool:
        return all([
            self.burst_pass,
            self.sustained_pass,
            self.ramp_pass,
            self.error_injection_pass,
            self.registry_pressure_pass,
        ])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ProductionReadinessReport:
    """
    Final production readiness report for the MedicalAIEngine platform.
    Contains all verification results, scores, and the GO / NO GO verdict.
    """
    # ── Identity ──────────────────────────────────────────────────────────────
    platform_name: str = "Medical Image Diagnosis & Anomaly Detection"
    platform_version: str = "0.14.2.7"
    report_timestamp: str = field(
        default_factory=lambda: datetime.now(tz=timezone.utc).isoformat()
    )

    # ── Verification Results ──────────────────────────────────────────────────
    verification_results: List[VerificationResult] = field(default_factory=list)

    # ── Quality Scores ────────────────────────────────────────────────────────
    quality_scores: QualityScores = field(default_factory=QualityScores)

    # ── Performance ───────────────────────────────────────────────────────────
    performance_summary: PerformanceSummary = field(default_factory=PerformanceSummary)

    # ── Stress ────────────────────────────────────────────────────────────────
    stress_summary: StressSummary = field(default_factory=StressSummary)

    # ── Health ────────────────────────────────────────────────────────────────
    health_score: float = 0.0
    health_status: str = "UNKNOWN"
    system_stats: Dict[str, Any] = field(default_factory=dict)

    # ── Verdict ───────────────────────────────────────────────────────────────
    verdict: ProductionVerdict = ProductionVerdict.NO_GO
    verdict_rationale: str = ""
    critical_failures: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    known_limitations: List[str] = field(default_factory=list)

    # ── Duration ──────────────────────────────────────────────────────────────
    total_verification_duration_seconds: float = 0.0

    # ── Computed ──────────────────────────────────────────────────────────────
    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.verification_results if r.passed)

    @property
    def failed_count(self) -> int:
        return sum(1 for r in self.verification_results if r.failed)

    @property
    def total_checks(self) -> int:
        return len(self.verification_results)

    def determine_verdict(self, pass_threshold: float = 75.0) -> ProductionVerdict:
        """Compute GO / NO GO based on quality scores and critical failures."""
        if self.critical_failures:
            self.verdict = ProductionVerdict.NO_GO
            self.verdict_rationale = f"Critical failures detected: {'; '.join(self.critical_failures[:3])}"
            return self.verdict

        overall = self.quality_scores.overall
        if overall >= pass_threshold and self.health_score >= 60.0:
            self.verdict = ProductionVerdict.GO
            self.verdict_rationale = (
                f"All critical checks passed. Overall quality score: {overall:.1f}/100. "
                f"Health score: {self.health_score:.1f}/100."
            )
        elif overall >= (pass_threshold - 10.0):
            self.verdict = ProductionVerdict.CONDITIONAL_GO
            self.verdict_rationale = (
                f"Quality score {overall:.1f}/100 is below threshold {pass_threshold}. "
                "Proceed with caution and address warnings."
            )
        else:
            self.verdict = ProductionVerdict.NO_GO
            self.verdict_rationale = (
                f"Overall quality score {overall:.1f}/100 below minimum threshold {pass_threshold}."
            )
        return self.verdict

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform_name": self.platform_name,
            "platform_version": self.platform_version,
            "report_timestamp": self.report_timestamp,
            "verdict": self.verdict.value,
            "verdict_rationale": self.verdict_rationale,
            "quality_scores": self.quality_scores.to_dict(),
            "health_score": self.health_score,
            "health_status": self.health_status,
            "performance_summary": self.performance_summary.to_dict(),
            "stress_summary": self.stress_summary.to_dict(),
            "verification_results": [r.to_dict() for r in self.verification_results],
            "critical_failures": self.critical_failures,
            "recommendations": self.recommendations,
            "known_limitations": self.known_limitations,
            "total_checks": self.total_checks,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_verification_duration_seconds": self.total_verification_duration_seconds,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def to_markdown(self) -> str:
        """Render the full production readiness report as a Markdown document."""
        ts = self.report_timestamp[:19].replace("T", " ") + " UTC"
        verdict_icon = {"GO": "✅", "NO GO": "❌", "CONDITIONAL GO": "⚠️"}.get(
            self.verdict.value, "?"
        )

        lines = [
            f"# Production Readiness Report — {self.platform_name}",
            f"**Version:** `{self.platform_version}` | **Generated:** {ts}",
            "",
            "---",
            "",
            "## Executive Summary",
            "",
            f"| Item | Value |",
            f"|------|-------|",
            f"| Platform | {self.platform_name} |",
            f"| Version | `{self.platform_version}` |",
            f"| Total Checks | {self.total_checks} |",
            f"| Passed | {self.passed_count} |",
            f"| Failed | {self.failed_count} |",
            f"| Overall Quality Score | **{self.quality_scores.overall:.1f}/100** ({self.quality_scores.grade()}) |",
            f"| Health Score | **{self.health_score:.1f}/100** |",
            f"| Health Status | `{self.health_status}` |",
            f"| Verification Duration | {self.total_verification_duration_seconds:.1f}s |",
            f"| **Verdict** | {verdict_icon} **{self.verdict.value}** |",
            "",
            "---",
            "",
            "## Quality Scores",
            "",
            "```",
            self.quality_scores.to_table(),
            "```",
            "",
            "---",
            "",
            "## Verification Results",
            "",
        ]

        for r in self.verification_results:
            lines.append(r.summary_line())

        lines += [
            "",
            "---",
            "",
            "## Performance Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Avg Latency | {self.performance_summary.avg_latency_ms:.1f} ms |",
            f"| Min Latency | {self.performance_summary.min_latency_ms:.1f} ms |",
            f"| Max Latency | {self.performance_summary.max_latency_ms:.1f} ms |",
            f"| p50 Latency | {self.performance_summary.p50_latency_ms:.1f} ms |",
            f"| p95 Latency | {self.performance_summary.p95_latency_ms:.1f} ms |",
            f"| p99 Latency | {self.performance_summary.p99_latency_ms:.1f} ms |",
            f"| Throughput | {self.performance_summary.throughput_rps:.2f} RPS |",
            f"| Peak Memory | {self.performance_summary.peak_memory_mb:.1f} MB |",
            "",
            "---",
            "",
            "## Stress Test Summary",
            "",
            f"| Test | Result |",
            f"|------|--------|",
            f"| Burst ({self.stress_summary.max_concurrency_achieved} threads) | {'PASS' if self.stress_summary.burst_pass else 'FAIL'} |",
            f"| Sustained Load | {'PASS' if self.stress_summary.sustained_pass else 'FAIL'} |",
            f"| Ramp-Up | {'PASS' if self.stress_summary.ramp_pass else 'FAIL'} |",
            f"| Error Injection | {'PASS' if self.stress_summary.error_injection_pass else 'FAIL'} |",
            f"| Registry Pressure | {'PASS' if self.stress_summary.registry_pressure_pass else 'FAIL'} |",
            f"| Total Requests | {self.stress_summary.total_requests} |",
            f"| Total Failures | {self.stress_summary.total_failures} |",
            f"| Failure Rate | {self.stress_summary.failure_rate * 100:.1f}% |",
            "",
            "---",
            "",
        ]

        if self.critical_failures:
            lines += ["## Critical Failures", ""]
            for cf in self.critical_failures:
                lines.append(f"- ❌ {cf}")
            lines.append("")
            lines.append("---")
            lines.append("")

        if self.recommendations:
            lines += ["## Recommendations", ""]
            for rec in self.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
            lines.append("---")
            lines.append("")

        if self.known_limitations:
            lines += ["## Known Limitations", ""]
            for lim in self.known_limitations:
                lines.append(f"- {lim}")
            lines.append("")
            lines.append("---")
            lines.append("")

        lines += [
            "## Final Verdict",
            "",
            f"> ## {verdict_icon} {self.verdict.value}",
            ">",
            f"> {self.verdict_rationale}",
            "",
        ]

        return "\n".join(lines)
