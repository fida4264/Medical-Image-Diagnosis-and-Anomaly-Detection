"""
Medical AI Engine — Verification Runner

Module 14.2.7: Production Verification & Stress Testing

Standalone orchestration entry point. Runs the full production
verification suite and writes production_readiness_report.md to disk.

Usage:
    python -m medical_ai_engine.verification.verification_runner
    python run_production_verification.py
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.production_validator import ProductionValidator
from medical_ai_engine.verification.verification_report import ProductionVerdict

# Ensure UTF-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("VerificationRunner")


def _print_header() -> None:
    print("=" * 70)
    print("  Medical AI Engine — Module 14.2.7")
    print("  Production Verification & Stress Testing")
    print("=" * 70)
    print()


def _print_progress(step: int, total: int, name: str) -> None:
    bar_len = 40
    filled = int(bar_len * step / total)
    bar = "[" + "#" * filled + " " * (bar_len - filled) + "]"
    print(f"  {bar}  {step}/{total}  {name}")


def run_verification(
    config: Optional[VerificationConfiguration] = None,
    report_path: Optional[str] = None,
) -> int:
    """
    Run the full production verification suite.

    Returns:
        0 — GO (passed)
        1 — NO GO / CONDITIONAL GO (failed or needs attention)
    """
    _print_header()
    cfg = config or VerificationConfiguration.fast()

    print(f"  Configuration Profile : {'fast (CI mode)' if cfg.benchmark_iterations <= 20 else 'full'}")
    print(f"  Supported Modalities  : {len(cfg.supported_modalities)}")
    print(f"  Thread Levels         : {cfg.thread_concurrency_levels}")
    print(f"  Benchmark Iterations  : {cfg.benchmark_iterations}")
    print(f"  Memory Cycles         : {cfg.memory_test_cycles}")
    print()

    start = time.perf_counter()

    validator = ProductionValidator(config=cfg)

    print("  Starting verification suite...\n")
    report = validator.run()

    # ── Print results ──────────────────────────────────────────────────────
    duration = time.perf_counter() - start
    print()
    print("=" * 70)
    print("  VERIFICATION RESULTS")
    print("=" * 70)

    for result in report.verification_results:
        icon = {
            "PASSED": "[PASS]",
            "FAILED": "[FAIL]",
            "WARNING": "[WARN]",
            "SKIPPED": "[SKIP]",
            "ERROR": "[ERR ]",
        }.get(result.status.value, "[????]")
        print(f"  {icon}  {result.name:<42}  score={result.score:5.1f}/100  ({result.duration_seconds:.1f}s)")

    print()
    print("=" * 70)
    print("  QUALITY SCORES")
    print("=" * 70)
    print()
    print(report.quality_scores.to_table())
    print()
    print("=" * 70)
    print("  SYSTEM HEALTH")
    print("=" * 70)
    print(f"  Health Score : {report.health_score:.1f}/100")
    print(f"  Status       : {report.health_status}")
    print(f"  Loaded Models: {report.system_stats.get('loaded_models', 'N/A')}")
    print(f"  Success Rate : {report.system_stats.get('success_rate', 0.0):.1%}")
    print(f"  Avg Latency  : {report.system_stats.get('avg_latency_ms', 0.0):.1f} ms")
    print()
    print("=" * 70)
    print("  PERFORMANCE SUMMARY")
    print("=" * 70)
    ps = report.performance_summary
    print(f"  Avg Latency  : {ps.avg_latency_ms:.1f} ms")
    print(f"  p95 Latency  : {ps.p95_latency_ms:.1f} ms")
    print(f"  p99 Latency  : {ps.p99_latency_ms:.1f} ms")
    print(f"  Throughput   : {ps.throughput_rps:.2f} RPS")
    print(f"  Peak Memory  : {ps.peak_memory_mb:.1f} MB")
    print()

    if report.recommendations:
        print("=" * 70)
        print("  RECOMMENDATIONS")
        print("=" * 70)
        for rec in report.recommendations:
            print(f"  - {rec}")
        print()

    verdict_icons = {
        "GO": "[ GO     ]",
        "NO GO": "[ NO GO  ]",
        "CONDITIONAL GO": "[ COND   ]",
    }
    print("=" * 70)
    verdict_display = verdict_icons.get(report.verdict.value, "[???????]")
    print(f"  FINAL VERDICT : {verdict_display}  {report.verdict.value}")
    print(f"  Rationale     : {report.verdict_rationale}")
    print(f"  Duration      : {duration:.1f}s")
    print("=" * 70)
    print()

    # ── Write Markdown report ──────────────────────────────────────────────
    output_path = Path(report_path or cfg.report_output_path)
    try:
        output_path.write_text(report.to_markdown(), encoding="utf-8")
        print(f"  Report saved to: {output_path.absolute()}")
    except Exception as e:
        logger.warning("Could not write report to disk: %s", e)

    print()

    return 0 if report.verdict == ProductionVerdict.GO else 1


if __name__ == "__main__":
    sys.exit(run_verification())
