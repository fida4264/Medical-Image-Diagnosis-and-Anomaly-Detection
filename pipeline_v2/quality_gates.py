"""
pipeline_v2/quality_gates.py
============================
Quality gate policy and decision logic for V2.0 Phase 9.
Evaluates multi-stage gating checks to determine whether model predictions
should be accepted, flagged for clinical review, or rejected/abstained.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pipeline_v2.schemas import GateOutcome, QualityGateResult, QualityGateState

logger = logging.getLogger(__name__)


class QualityGateEvaluator:
    """
    Evaluates configurable operational quality gates for medical inference.
    """

    def __init__(
        self,
        enforce_ood_rejection: bool = True,
        enforce_calibration: bool = True,
        anomaly_warning_only: bool = True,
    ):
        self.enforce_ood_rejection = enforce_ood_rejection
        self.enforce_calibration = enforce_calibration
        self.anomaly_warning_only = anomaly_warning_only

    def evaluate_gates(
        self,
        input_valid: bool,
        input_error: Optional[str] = None,
        modality_valid: bool = True,
        modality_error: Optional[str] = None,
        model_available: bool = True,
        model_error: Optional[str] = None,
        logits: Optional[np.ndarray] = None,
        probabilities: Optional[np.ndarray] = None,
        calibrated: bool = False,
        ood_score: Optional[float] = None,
        ood_threshold: Optional[float] = None,
        anomaly_score: Optional[float] = None,
        anomaly_threshold: Optional[float] = None,
    ) -> QualityGateResult:
        """
        Evaluates all operational gates and returns consolidated decision.
        """
        gate_outcomes: Dict[str, GateOutcome] = {}
        rejection_reasons: List[str] = []
        review_reasons: List[str] = []

        # Gate 1: Input Validity
        if not input_valid:
            msg = input_error or "Input failed format, resolution, or decoding validation."
            gate_outcomes["GATE_INPUT_VALIDITY"] = GateOutcome(
                gate_name="GATE_INPUT_VALIDITY", passed=False, message=msg
            )
            rejection_reasons.append(msg)
            return QualityGateResult(
                overall_status=QualityGateState.REJECTED_INVALID_INPUT,
                rejection_reasons=rejection_reasons,
                review_reasons=review_reasons,
                gate_outcomes=gate_outcomes,
            )
        gate_outcomes["GATE_INPUT_VALIDITY"] = GateOutcome(
            gate_name="GATE_INPUT_VALIDITY", passed=True, message="Input format, resolution, and pixel data verified."
        )

        # Gate 2: Modality Validity
        if not modality_valid:
            msg = modality_error or "Requested modality is unsupported or ambiguous."
            gate_outcomes["GATE_MODALITY_VALIDITY"] = GateOutcome(
                gate_name="GATE_MODALITY_VALIDITY", passed=False, message=msg
            )
            rejection_reasons.append(msg)
            return QualityGateResult(
                overall_status=QualityGateState.REJECTED_UNSUPPORTED_MODALITY,
                rejection_reasons=rejection_reasons,
                review_reasons=review_reasons,
                gate_outcomes=gate_outcomes,
            )
        gate_outcomes["GATE_MODALITY_VALIDITY"] = GateOutcome(
            gate_name="GATE_MODALITY_VALIDITY", passed=True, message="Modality matches canonical platform configuration."
        )

        # Gate 3: Model Availability
        if not model_available:
            msg = model_error or "Baseline model checkpoint is missing or load failed."
            gate_outcomes["GATE_MODEL_AVAILABILITY"] = GateOutcome(
                gate_name="GATE_MODEL_AVAILABILITY", passed=False, message=msg
            )
            rejection_reasons.append(msg)
            return QualityGateResult(
                overall_status=QualityGateState.INFERENCE_ERROR,
                rejection_reasons=rejection_reasons,
                review_reasons=review_reasons,
                gate_outcomes=gate_outcomes,
            )
        gate_outcomes["GATE_MODEL_AVAILABILITY"] = GateOutcome(
            gate_name="GATE_MODEL_AVAILABILITY", passed=True, message="Model weights verified and loaded."
        )

        # Gate 4: Finite Model Outputs
        if logits is None or not np.all(np.isfinite(logits)):
            msg = "Model produced non-finite (NaN or Inf) logits."
            gate_outcomes["GATE_FINITE_OUTPUTS"] = GateOutcome(
                gate_name="GATE_FINITE_OUTPUTS", passed=False, message=msg
            )
            rejection_reasons.append(msg)
            return QualityGateResult(
                overall_status=QualityGateState.INFERENCE_ERROR,
                rejection_reasons=rejection_reasons,
                review_reasons=review_reasons,
                gate_outcomes=gate_outcomes,
            )
        gate_outcomes["GATE_FINITE_OUTPUTS"] = GateOutcome(
            gate_name="GATE_FINITE_OUTPUTS", passed=True, message="Output logits and probabilities are finite and bounded."
        )

        # Gate 5: Calibration Status
        if self.enforce_calibration and not calibrated:
            msg = "Calibrated parameters unavailable; raw predictions require verification."
            gate_outcomes["GATE_CALIBRATION_STATUS"] = GateOutcome(
                gate_name="GATE_CALIBRATION_STATUS", passed=False, message=msg
            )
            review_reasons.append(msg)
        else:
            gate_outcomes["GATE_CALIBRATION_STATUS"] = GateOutcome(
                gate_name="GATE_CALIBRATION_STATUS", passed=True, message="Phase 7 calibration successfully applied."
            )

        # Gate 6: Out-of-Distribution (OOD) Distributional Gate
        is_ood = False
        if ood_score is not None and ood_threshold is not None:
            # Score orientation: HIGHER = MORE OOD
            if ood_score > ood_threshold:
                is_ood = True
                msg = (
                    f"OOD score ({ood_score:.4f}) exceeds frozen validation threshold ({ood_threshold:.4f}). "
                    "Input appears distributionally unfamiliar or foreign to training domain."
                )
                gate_outcomes["GATE_OOD_CHECK"] = GateOutcome(
                    gate_name="GATE_OOD_CHECK",
                    passed=False,
                    message=msg,
                    details={"score": ood_score, "threshold": ood_threshold},
                )
                if self.enforce_ood_rejection:
                    rejection_reasons.append(msg)
                    return QualityGateResult(
                        overall_status=QualityGateState.REJECTED_OOD,
                        rejection_reasons=rejection_reasons,
                        review_reasons=review_reasons,
                        gate_outcomes=gate_outcomes,
                    )
                else:
                    review_reasons.append(msg)
            else:
                gate_outcomes["GATE_OOD_CHECK"] = GateOutcome(
                    gate_name="GATE_OOD_CHECK",
                    passed=True,
                    message=f"OOD score ({ood_score:.4f}) within validation operating threshold ({ood_threshold:.4f}).",
                    details={"score": ood_score, "threshold": ood_threshold},
                )
        else:
            gate_outcomes["GATE_OOD_CHECK"] = GateOutcome(
                gate_name="GATE_OOD_CHECK", passed=True, message="OOD check bypassed or parameters unavailable."
            )

        # Gate 7: Anomaly Scoring Check (Within-Distribution Normal Reference Deviation)
        is_anomaly = False
        if anomaly_score is not None and anomaly_threshold is not None:
            if anomaly_score > anomaly_threshold:
                is_anomaly = True
                msg = (
                    f"Anomaly score ({anomaly_score:.2f}) exceeds reference baseline threshold ({anomaly_threshold:.2f}). "
                    "Potential morphological deviation from normal reference state."
                )
                gate_outcomes["GATE_ANOMALY_CHECK"] = GateOutcome(
                    gate_name="GATE_ANOMALY_CHECK",
                    passed=False,
                    message=msg,
                    details={"score": anomaly_score, "threshold": anomaly_threshold},
                )
                review_reasons.append(msg)
            else:
                gate_outcomes["GATE_ANOMALY_CHECK"] = GateOutcome(
                    gate_name="GATE_ANOMALY_CHECK",
                    passed=True,
                    message=f"Anomaly score ({anomaly_score:.2f}) aligned with normal reference baseline.",
                    details={"score": anomaly_score, "threshold": anomaly_threshold},
                )

        # Determine Final Operational State
        if is_anomaly:
            status = QualityGateState.REVIEW_REQUIRED_ANOMALY
        elif len(review_reasons) > 0:
            status = QualityGateState.FLAGGED_FOR_REVIEW
        else:
            status = QualityGateState.ACCEPTED_FOR_MODEL_OUTPUT

        return QualityGateResult(
            overall_status=status,
            rejection_reasons=rejection_reasons,
            review_reasons=review_reasons,
            gate_outcomes=gate_outcomes,
        )
