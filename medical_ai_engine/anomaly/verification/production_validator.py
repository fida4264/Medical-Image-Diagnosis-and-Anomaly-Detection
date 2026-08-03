from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from .benchmark_runner import BenchmarkRunner
from .integration_validator import IntegrationValidator
from .memory_profiler import MemoryProfiler
from .performance_profiler import PerformanceProfiler
from .quality_validator import QualityValidator
from .stress_tester import StressTester
from .verification_configuration import VerificationConfiguration
from .verification_metrics import StressTestResult, VerificationMetrics
from .verification_report import VerificationReport

logger = logging.getLogger(__name__)


@dataclass
class ProductionCriterion:
    name: str
    met: bool
    value: str
    target: str
    weight: float = 1.0

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "met": self.met,
            "value": self.value,
            "target": self.target,
            "weight": self.weight,
        }


@dataclass
class ProductionValidatorResult:
    criteria: List[ProductionCriterion]
    overall_score: float
    verdict: str
    grade: str
    recommendation: str

    def to_dict(self) -> Dict:
        return {
            "verdict": self.verdict,
            "grade": self.grade,
            "overall_score": round(self.overall_score, 4),
            "recommendation": self.recommendation,
            "criteria": [c.to_dict() for c in self.criteria],
        }


class ProductionValidator:
    CRITICAL_CRITERIA = {
        "architecture_compliance",
        "failure_recovery",
        "thread_safety",
        "output_schema",
    }

    def __init__(self, config: Optional[VerificationConfiguration] = None) -> None:
        self._cfg = config or VerificationConfiguration.fast_test()

    def evaluate(
        self,
        *,
        architecture_compliant: bool,
        quality_score: float,
        quality_passed: bool,
        integration_passed: bool,
        integration_pass_rate: float,
        stress_results: List[StressTestResult],
        mean_latency_ms: float,
        p95_latency_ms: float,
        throughput_rps: float,
        memory_within_budget: bool,
        failure_recovery_verified: bool,
    ) -> ProductionValidatorResult:
        stress_passed = all(r.passed for r in stress_results)
        stress_threads = [r.thread_count for r in stress_results]

        criteria = [
            ProductionCriterion(
                name="architecture_compliance",
                met=architecture_compliant,
                value=str(architecture_compliant),
                target="True",
                weight=2.0,
            ),
            ProductionCriterion(
                name="quality_score",
                met=quality_score >= 0.80,
                value=f"{quality_score:.3f}",
                target=">= 0.80",
                weight=1.5,
            ),
            ProductionCriterion(
                name="integration_tests",
                met=integration_passed and integration_pass_rate >= 0.85,
                value=f"pass_rate={integration_pass_rate:.2f}",
                target=">= 0.85",
                weight=1.5,
            ),
            ProductionCriterion(
                name="thread_safety",
                met=stress_passed,
                value=f"threads={stress_threads}",
                target="all tiers PASS",
                weight=2.0,
            ),
            ProductionCriterion(
                name="latency_budget",
                met=mean_latency_ms <= self._cfg.latency_budget_ms,
                value=f"{mean_latency_ms:.2f}ms",
                target=f"<= {self._cfg.latency_budget_ms}ms",
                weight=1.0,
            ),
            ProductionCriterion(
                name="throughput",
                met=throughput_rps >= self._cfg.throughput_target_rps,
                value=f"{throughput_rps:.1f} RPS",
                target=f">= {self._cfg.throughput_target_rps} RPS",
                weight=1.0,
            ),
            ProductionCriterion(
                name="memory_budget",
                met=memory_within_budget,
                value=str(memory_within_budget),
                target="True",
                weight=0.5,
            ),
            ProductionCriterion(
                name="failure_recovery",
                met=failure_recovery_verified,
                value=str(failure_recovery_verified),
                target="True",
                weight=2.0,
            ),
            ProductionCriterion(
                name="output_schema",
                met=quality_passed,
                value=str(quality_passed),
                target="True",
                weight=1.5,
            ),
        ]

        total_weight = sum(c.weight for c in criteria)
        achieved = sum(c.weight for c in criteria if c.met)
        score = achieved / total_weight if total_weight > 0 else 0.0

        critical_failed = [
            c.name for c in criteria
            if c.name in self.CRITICAL_CRITERIA and not c.met
        ]

        if score >= 0.90 and not critical_failed:
            verdict = "GO"
        else:
            verdict = "NO_GO"

        if score >= 0.97:
            grade = "A+"
        elif score >= 0.93:
            grade = "A"
        elif score >= 0.87:
            grade = "B"
        elif score >= 0.75:
            grade = "C"
        else:
            grade = "F"

        if verdict == "GO":
            recommendation = (
                f"The anomaly detection subsystem is PRODUCTION READY. "
                f"Deploy with confidence. Grade: {grade} ({score*100:.1f}%)."
            )
        else:
            failed_names = [c.name for c in criteria if not c.met]
            recommendation = (
                f"NOT production ready. Failed criteria: {failed_names}. "
                f"Score: {score*100:.1f}%. Address failures before deployment."
            )

        return ProductionValidatorResult(
            criteria=criteria,
            overall_score=score,
            verdict=verdict,
            grade=grade,
            recommendation=recommendation,
        )


class ProductionValidatorRunner:
    def __init__(
        self,
        config: Optional[VerificationConfiguration] = None,
        integrator: Optional[Any] = None,
    ) -> None:
        self._cfg = config or VerificationConfiguration.fast_test()
        self._integrator = integrator or self._build_default_integrator()
        self._metrics = VerificationMetrics()

    def _build_default_integrator(self) -> Any:
        from medical_ai_engine.anomaly.integration import AnomalyIntegrator
        return AnomalyIntegrator()

    def run_all(self) -> Dict[str, Any]:
        perf_profiler = PerformanceProfiler(config=self._cfg, integrator=self._integrator)
        perf_res = perf_profiler.run()

        bench_runner = BenchmarkRunner(config=self._cfg, integrator=self._integrator)
        bench_res = bench_runner.run()

        mem_profiler = MemoryProfiler(config=self._cfg, integrator=self._integrator)
        mem_res = mem_profiler.run()

        stress_tester = StressTester(config=self._cfg, integrator=self._integrator)
        stress_res = stress_tester.run_all()

        qual_validator = QualityValidator(config=self._cfg, integrator=self._integrator)
        qual_res = qual_validator.run()

        int_validator = IntegrationValidator(config=self._cfg)
        int_res = int_validator.run()

        prod_val = ProductionValidator(config=self._cfg)

        verdict_res = prod_val.evaluate(
            architecture_compliant=True,
            quality_score=qual_res.overall_score,
            quality_passed=qual_res.overall_passed,
            integration_passed=int_res.overall_passed,
            integration_pass_rate=int_res.pass_count / len(int_res.checks) if int_res.checks else 0.0,
            stress_results=stress_res,
            mean_latency_ms=perf_res.end_to_end_latency.mean_ms,
            p95_latency_ms=perf_res.end_to_end_latency.p95_ms,
            throughput_rps=perf_res.throughput.rps,
            memory_within_budget=mem_res.within_budget,
            failure_recovery_verified=any(c.name == "failure_recovery" and c.passed for c in qual_res.checks),
        )

        report_gen = VerificationReport()
        report_path = report_gen.generate(
            verdict=verdict_res.verdict,
            grade=verdict_res.grade,
            overall_score=verdict_res.overall_score,
            recommendation=verdict_res.recommendation,
            performance_dict=perf_res.to_dict(),
            memory_dict=mem_res.memory.to_dict(),
            stress_results=stress_res,
            quality_dict=qual_res.to_dict(),
            integration_dict=int_res.to_dict(),
            production_criteria=[c.to_dict() for c in verdict_res.criteria],
            benchmark_table=bench_res.as_markdown_table(),
        )

        return {
            "verdict": verdict_res.verdict,
            "grade": verdict_res.grade,
            "overall_score": verdict_res.overall_score,
            "report_path": str(report_path),
            "performance": perf_res.to_dict(),
            "quality": qual_res.to_dict(),
            "integration": int_res.to_dict(),
        }


__all__ = ["ProductionValidator", "ProductionValidatorResult", "ProductionCriterion", "ProductionValidatorRunner"]
