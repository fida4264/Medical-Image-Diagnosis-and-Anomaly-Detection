"""
Medical AI Engine Core — Test Runner Script

Success Criteria Demonstrator:
  1. Creates EngineContext
  2. Creates ExecutionSession
  3. Initializes MedicalAIEngine
  4. Transitions through lifecycle (CREATED -> INITIALIZED -> RUNNING -> COMPLETED)
  5. Generates EngineStatistics
  6. Publishes Engine Events
  7. Logs Execution to console & logs/medical_ai_engine.log
  8. Runs pytest unit test suite

All achieved WITHOUT importing or executing any AI modality model or routing logic.
"""

import sys
from pathlib import Path
import pytest

from medical_ai_engine.core.engine_configuration import EngineConfiguration
from medical_ai_engine.core.engine_context import EngineContext
from medical_ai_engine.core.engine_state import EngineState
from medical_ai_engine.core.execution_session import ExecutionSession
from medical_ai_engine.core.medical_ai_engine import MedicalAIEngine
from medical_ai_engine.interfaces.pipeline_stage import BasePipelineStage


class SampleOperatingSystemStage(BasePipelineStage):
    """Sample pipeline stage demonstrating interface invocation."""

    def __init__(self, stage_name: str = "SampleOSStage") -> None:
        self._name = stage_name

    @property
    def name(self) -> str:
        return self._name

    def validate(self, ctx: EngineContext) -> None:
        ctx.log(f"Pre-condition validation passed for {self._name}", stage=self._name)

    def process(self, ctx: EngineContext) -> None:
        ctx.metadata["operating_system_status"] = "ACTIVE"
        ctx.metadata["framework_version"] = "0.14.2.1"
        ctx.log("Operating system pipeline stage processed.", stage=self._name)


def main():
    print("==================================================================")
    print("      Module 14.2.1: MedicalAIEngine Core Verification Runner      ")
    print("==================================================================")

    # 1. Configuration
    config = EngineConfiguration(
        engine_name="MedicalAIEngineCore",
        log_file_path="logs/medical_ai_engine.log",
        enable_console_logging=True,
    )

    # 2. Stage instantiation
    stage = SampleOperatingSystemStage()

    # 3. Create Engine
    engine = MedicalAIEngine(config=config, pipeline_stages=[stage])
    print(f"[1/5] Engine created. State: {engine.state.value}")

    # 4. Initialize Engine
    engine.initialize()
    print(f"[2/5] Engine initialized. State: {engine.state.value}")

    # 5. Execute Request
    print("[3/5] Executing sample request through engine core...")
    ctx = engine.execute(
        image_source="sample_patient_image.dcm",
        manual_modality_override="brain_mri",
        extra_metadata={"patient_id": "P-10492", "study_date": "2026-08-01"},
    )
    print(f"[4/5] Execution complete. Context State: {ctx.execution_state}")
    print(f"      Request ID:  {ctx.request_id}")
    print(f"      Session ID:  {ctx.session_id}")
    print(f"      Total Time:  {ctx.total_execution_ms():.2f} ms")
    print(f"      Logs Count:  {len(ctx.logs)}")

    # 6. Print Statistics
    stats = engine.get_statistics()
    print(f"[5/5] Operational Statistics:\n      {stats}")

    health = engine.get_health_status()
    print(f"      Health Status: {health['status']}")

    engine.shutdown()
    print("\n------------------------------------------------------------------")
    print("Running PyTest Suite for medical_ai_engine...")
    print("------------------------------------------------------------------")

    pytest_args = ["medical_ai_engine/tests/test_engine_core.py", "-v"]
    ret = pytest.main(pytest_args)

    if ret == 0:
        print("\n[SUCCESS] All Module 14.2.1 Engine Core tests PASSED successfully!")
    else:
        print(f"\n[FAILED] PyTest failed with exit code: {ret}")
        sys.exit(ret)


if __name__ == "__main__":
    main()
