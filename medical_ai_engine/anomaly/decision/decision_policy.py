"""
decision_policy.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Configurable decision policies for signal fusion.
"""
from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

from medical_ai_engine.anomaly.anomaly_result import AnomalyStatus, RecommendedAction, RiskLevel
from .decision_configuration import DecisionConfiguration
from .decision_exceptions import PolicyError
from .priority_manager import PriorityManager, SignalType

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionOutput:
    """
    Standardized intermediate output produced by a DecisionPolicy.
    """
    status: AnomalyStatus
    known_unknown_status: str
    risk_level: RiskLevel
    risk_score: float
    recommended_action: RecommendedAction
    reason: str
    decision_confidence: float
    primary_signal: str


class IDecisionPolicy(abc.ABC):
    """Abstract interface for decision fusion policies."""

    @abc.abstractmethod
    def evaluate(
        self,
        signals: Dict[str, Any],
        confidence: float,
        modality: str,
    ) -> DecisionOutput:
        """
        Evaluate inputs signals and return a DecisionOutput.

        Parameters
        ----------
        signals : Dict[str, Any]
            Map of signal_name -> signal payload dictionary.
        confidence : float
            Original prediction confidence score.
        modality : str
            Imaging modality name.
        """


class DefaultPriorityPolicy(IDecisionPolicy):
    """
    Default Priority Decision Policy.

    Evaluates signals in priority order:
    1. Unknown Disease → Highest Priority → ESCALATE / HIGH Risk
    2. OOD             → Second Priority → ESCALATE / HIGH Risk
    3. Low Confidence  → Third Priority  → MANUAL_REVIEW / MODERATE-HIGH Risk
    4. Poor Quality    → Fourth Priority → REPEAT_SCAN / MODERATE Risk
    5. Known Disease   → Default Baseline → PROCEED / LOW Risk
    """

    def __init__(
        self,
        config: Optional[DecisionConfiguration] = None,
        priority_manager: Optional[PriorityManager] = None,
    ) -> None:
        self._cfg = config or DecisionConfiguration.default()
        self._pm = priority_manager or PriorityManager(self._cfg.priority_order)

    def evaluate(
        self,
        signals: Dict[str, Any],
        confidence: float,
        modality: str,
    ) -> DecisionOutput:
        active_signals = []
        if signals.get("unknown_disease", {}).get("is_unknown", False):
            active_signals.append(SignalType.UNKNOWN_DISEASE)
        if signals.get("ood", {}).get("is_ood", False):
            active_signals.append(SignalType.OOD)
        if signals.get("low_confidence", {}).get("is_low_confidence", False):
            active_signals.append(SignalType.LOW_CONFIDENCE)
        if signals.get("poor_quality", {}).get("is_poor_quality", False):
            active_signals.append(SignalType.POOR_QUALITY)

        primary = self._pm.resolve_highest_priority(active_signals)

        if primary == SignalType.UNKNOWN_DISEASE:
            unknown_info = signals.get("unknown_disease", {})
            novelty = float(unknown_info.get("novelty_score", 0.80))
            return DecisionOutput(
                status=AnomalyStatus.UNKNOWN_DISEASE,
                known_unknown_status="UNKNOWN",
                risk_level=RiskLevel.HIGH,
                risk_score=novelty,
                recommended_action=RecommendedAction.ESCALATE,
                reason=f"Unknown disease pattern detected (novelty score: {novelty:.3f}).",
                decision_confidence=min(1.0, 0.5 + novelty / 2.0),
                primary_signal=primary,
            )

        if primary == SignalType.OOD:
            ood_info = signals.get("ood", {})
            score = float(ood_info.get("ood_score", 0.85))
            return DecisionOutput(
                status=AnomalyStatus.OUT_OF_DISTRIBUTION,
                known_unknown_status="UNKNOWN",
                risk_level=RiskLevel.HIGH,
                risk_score=min(1.0, score / 10.0 if score > 1 else score),
                recommended_action=RecommendedAction.ESCALATE,
                reason=f"Out-of-distribution sample detected (OOD distance: {score:.3f}).",
                decision_confidence=0.90,
                primary_signal=primary,
            )

        if primary == SignalType.LOW_CONFIDENCE:
            risk = RiskLevel.HIGH if confidence < 0.35 else RiskLevel.MODERATE
            action = RecommendedAction.ESCALATE if confidence < 0.35 else RecommendedAction.MANUAL_REVIEW
            return DecisionOutput(
                status=AnomalyStatus.LOW_CONFIDENCE,
                known_unknown_status="UNCERTAIN",
                risk_level=risk,
                risk_score=round(1.0 - confidence, 4),
                recommended_action=action,
                reason=f"Prediction confidence {confidence:.3f} below clinical threshold.",
                decision_confidence=confidence,
                primary_signal=primary,
            )

        if primary == SignalType.POOR_QUALITY:
            return DecisionOutput(
                status=AnomalyStatus.POOR_QUALITY,
                known_unknown_status="UNCERTAIN",
                risk_level=RiskLevel.MODERATE,
                risk_score=0.60,
                recommended_action=RecommendedAction.REPEAT_SCAN,
                reason="Image quality insufficient for reliable diagnosis.",
                decision_confidence=0.80,
                primary_signal=primary,
            )

        # Baseline: KNOWN_DISEASE
        return DecisionOutput(
            status=AnomalyStatus.KNOWN_DISEASE,
            known_unknown_status="KNOWN",
            risk_level=RiskLevel.LOW,
            risk_score=round(1.0 - confidence, 4),
            recommended_action=RecommendedAction.PROCEED,
            reason="All anomaly checks passed; prediction is trusted.",
            decision_confidence=confidence,
            primary_signal=SignalType.KNOWN_DISEASE,
        )


class WeightedFusionPolicy(IDecisionPolicy):
    """
    Weighted Fusion Decision Policy.

    Fuses signals linearly:
        Risk = w_unk * S_unk + w_ood * S_ood + w_conf * (1 - confidence)
    """

    def __init__(self, config: Optional[DecisionConfiguration] = None) -> None:
        self._cfg = config or DecisionConfiguration.weighted()

    def evaluate(
        self,
        signals: Dict[str, Any],
        confidence: float,
        modality: str,
    ) -> DecisionOutput:
        s_unk = float(signals.get("unknown_disease", {}).get("novelty_score", 0.0))
        s_ood_raw = float(signals.get("ood", {}).get("ood_score", 0.0))
        s_ood = min(1.0, s_ood_raw / 10.0) if s_ood_raw > 1.0 else s_ood_raw
        s_conf = max(0.0, 1.0 - confidence)

        w_unk = self._cfg.unknown_weight
        w_ood = self._cfg.ood_weight
        w_conf = self._cfg.confidence_weight

        risk_score = min(1.0, w_unk * s_unk + w_ood * s_ood + w_conf * s_conf)

        th_high = self._cfg.per_modality_thresholds.get(modality, self._cfg.high_risk_threshold)
        th_crit = self._cfg.critical_risk_threshold

        if risk_score >= th_crit:
            return DecisionOutput(
                status=AnomalyStatus.UNKNOWN_DISEASE,
                known_unknown_status="UNKNOWN",
                risk_level=RiskLevel.CRITICAL,
                risk_score=risk_score,
                recommended_action=RecommendedAction.REJECT_PREDICTION,
                reason=f"Weighted risk score {risk_score:.3f} reached CRITICAL threshold ({th_crit}).",
                decision_confidence=risk_score,
                primary_signal="weighted_fusion_critical",
            )
        if risk_score >= th_high:
            return DecisionOutput(
                status=AnomalyStatus.UNKNOWN_DISEASE,
                known_unknown_status="UNKNOWN",
                risk_level=RiskLevel.HIGH,
                risk_score=risk_score,
                recommended_action=RecommendedAction.ESCALATE,
                reason=f"Weighted risk score {risk_score:.3f} reached HIGH threshold ({th_high}).",
                decision_confidence=risk_score,
                primary_signal="weighted_fusion_high",
            )
        if risk_score >= 0.40:
            return DecisionOutput(
                status=AnomalyStatus.LOW_CONFIDENCE,
                known_unknown_status="UNCERTAIN",
                risk_level=RiskLevel.MODERATE,
                risk_score=risk_score,
                recommended_action=RecommendedAction.MANUAL_REVIEW,
                reason=f"Weighted risk score {risk_score:.3f} indicates moderate uncertainty.",
                decision_confidence=1.0 - risk_score,
                primary_signal="weighted_fusion_moderate",
            )

        return DecisionOutput(
            status=AnomalyStatus.KNOWN_DISEASE,
            known_unknown_status="KNOWN",
            risk_level=RiskLevel.LOW,
            risk_score=risk_score,
            recommended_action=RecommendedAction.PROCEED,
            reason="Weighted risk score indicates trusted in-distribution sample.",
            decision_confidence=1.0 - risk_score,
            primary_signal="weighted_fusion_low",
        )


class PolicyRegistry:
    """Registry for policy registration and retrieval."""

    def __init__(self) -> None:
        self._policies: Dict[str, Type[IDecisionPolicy]] = {
            "default_priority": DefaultPriorityPolicy,
            "weighted_fusion": WeightedFusionPolicy,
        }

    def register(self, name: str, policy_cls: Type[IDecisionPolicy]) -> None:
        self._policies[name] = policy_cls

    def get(self, name: str, config: Optional[DecisionConfiguration] = None) -> IDecisionPolicy:
        if name not in self._policies:
            raise PolicyError(f"Unknown decision policy '{name}'. Available: {list(self._policies.keys())}")
        return self._policies[name](config=config)


__all__ = [
    "IDecisionPolicy",
    "DecisionOutput",
    "DefaultPriorityPolicy",
    "WeightedFusionPolicy",
    "PolicyRegistry",
]
