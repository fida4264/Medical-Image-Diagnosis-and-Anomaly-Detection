"""
run_verification.py
===================
CLI Entry point for Module 5.5: System Verification & Integration Testing.

Usage
-----
    python run_verification.py

Author: Medical AI System — Module 5.5
"""

import sys
from system_verification.runner import VerificationRunner

def main() -> int:
    runner = VerificationRunner()
    res = runner.run()
    decision = res.get("health_data", {}).get("readiness", {}).get("decision", "NO GO")
    return 0 if "GO" in decision else 1

if __name__ == "__main__":
    sys.exit(main())
