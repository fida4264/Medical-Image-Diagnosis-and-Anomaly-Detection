"""
priority_manager.py — Module 16.5: Risk Assessment & Decision Fusion
Medical AI Platform — Manages signal priorities and resolves conflicts between detection signals.
"""
from __future__ import annotations

import logging
from enum import IntEnum
from typing import Dict, List, Optional

from .decision_exceptions import PriorityError

logger = logging.getLogger(__name__)


class SignalType(str):
    """Detection signal types."""
    UNKNOWN_DISEASE = "unknown_disease"
    OOD = "ood"
    LOW_CONFIDENCE = "low_confidence"
    POOR_QUALITY = "poor_quality"
    KNOWN_DISEASE = "known_disease"


class PriorityManager:
    """
    Manages priority ordering for decision signals.

    Ensures that when multiple detection strategies fire simultaneously (e.g., both
    OOD and Unknown Disease are detected), the signal with the highest priority according
    to the active policy wins.

    Default Priority Hierarchy
    --------------------------
    1. Unknown Disease  (rank 1 — highest priority)
    2. OOD               (rank 2)
    3. Low Confidence    (rank 3)
    4. Poor Quality      (rank 4)
    5. Known Disease     (rank 5 — baseline)
    """

    def __init__(self, priority_order: Optional[List[str]] = None) -> None:
        default_order = [
            SignalType.UNKNOWN_DISEASE,
            SignalType.OOD,
            SignalType.LOW_CONFIDENCE,
            SignalType.POOR_QUALITY,
        ]
        self._order = priority_order or default_order
        self._rank_map: Dict[str, int] = {
            signal: idx + 1 for idx, signal in enumerate(self._order)
        }
        # Fallback rank for unmapped signals
        self._rank_map[SignalType.KNOWN_DISEASE] = 99
        logger.info("PriorityManager initialised with order: %s", self._order)

    def get_rank(self, signal: str) -> int:
        """Return numeric priority rank for *signal* (1 is highest priority)."""
        return self._rank_map.get(signal, 100)

    def resolve_highest_priority(self, active_signals: List[str]) -> str:
        """
        Given a list of active signals (signals that triggered an anomaly flag),
        return the signal with the highest priority (lowest rank).

        Parameters
        ----------
        active_signals : List[str]
            List of active signal keys.

        Returns
        -------
        str
            Highest priority signal name. Returns "known_disease" if active_signals is empty.
        """
        if not active_signals:
            return SignalType.KNOWN_DISEASE
        try:
            return min(active_signals, key=lambda s: self.get_rank(s))
        except Exception as exc:
            raise PriorityError(f"Failed to resolve highest priority signal: {exc}") from exc

    def sort_by_priority(self, signals: List[str]) -> List[str]:
        """Sort a list of signal keys by priority (highest priority first)."""
        return sorted(signals, key=lambda s: self.get_rank(s))

    @property
    def priority_order(self) -> List[str]:
        return list(self._order)


__all__ = ["PriorityManager", "SignalType"]
