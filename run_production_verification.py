"""
Module 14.2.7 — Production Verification & Stress Testing

Top-level convenience runner.

Usage:
    python run_production_verification.py          # fast profile (CI)
    python run_production_verification.py --full   # full profile
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
from medical_ai_engine.verification.verification_configuration import VerificationConfiguration
from medical_ai_engine.verification.verification_runner import run_verification


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Medical AI Engine — Module 14.2.7 Production Verification"
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run the full verification suite (slower, more thorough). Default: fast profile.",
    )
    parser.add_argument(
        "--report",
        default="production_readiness_report.md",
        help="Output path for the production readiness report (default: production_readiness_report.md).",
    )
    args = parser.parse_args()

    config = VerificationConfiguration.default() if args.full else VerificationConfiguration.fast()

    exit_code = run_verification(config=config, report_path=args.report)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
