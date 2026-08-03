"""
leaderboard.py — Module 17: Global Benchmarking & Model Comparison
Medical AI Platform — Overall platform leaderboard, per-modality tables, and ranking utilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from medical_ai_engine.benchmark.benchmark_exceptions import LeaderboardError
from medical_ai_engine.benchmark.comparison_engine import ComparisonSummary, ModelComparisonEntry

logger = logging.getLogger(__name__)


@dataclass
class LeaderboardTier:
    """A named performance tier bracket (e.g. EXCELLENT, GOOD, FAIR, POOR)."""

    name: str
    min_score: float
    max_score: float
    icon: str

    def contains(self, score: float) -> bool:
        return self.min_score <= score <= self.max_score


LEADERBOARD_TIERS = [
    LeaderboardTier("EXCELLENT", 0.90, 1.00, "🏆"),
    LeaderboardTier("GOOD",      0.75, 0.90, "✅"),
    LeaderboardTier("FAIR",      0.60, 0.75, "⚠️"),
    LeaderboardTier("POOR",      0.00, 0.60, "❌"),
]


def _get_tier(score: float) -> LeaderboardTier:
    for t in LEADERBOARD_TIERS:
        if t.contains(score):
            return t
    return LEADERBOARD_TIERS[-1]


@dataclass
class LeaderboardRow:
    """A single row in the platform leaderboard."""

    rank: int
    tier: str
    tier_icon: str
    modality: str
    model_name: str
    architecture: str
    composite_score: float
    f1_score: float
    accuracy: float
    roc_auc: Optional[float]
    inference_latency_ms: float
    throughput_rps: float
    efficiency_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "tier": self.tier,
            "tier_icon": self.tier_icon,
            "modality": self.modality,
            "model_name": self.model_name,
            "architecture": self.architecture,
            "composite_score": round(self.composite_score, 4),
            "f1_score": round(self.f1_score, 4),
            "accuracy": round(self.accuracy, 4),
            "roc_auc": round(self.roc_auc, 4) if self.roc_auc is not None else None,
            "inference_latency_ms": round(self.inference_latency_ms, 3),
            "throughput_rps": round(self.throughput_rps, 2),
            "efficiency_ratio": round(self.efficiency_ratio, 4),
        }

    def to_markdown_row(self) -> str:
        roc_str = f"{self.roc_auc:.4f}" if self.roc_auc is not None else "N/A"
        return (
            f"| {self.rank} | {self.tier_icon} {self.tier} | {self.modality} | "
            f"{self.model_name} | {self.architecture} | {self.composite_score:.4f} | "
            f"{self.f1_score:.4f} | {self.accuracy:.4f} | {roc_str} | "
            f"{self.inference_latency_ms:.2f} | {self.throughput_rps:.1f} |"
        )


@dataclass
class Leaderboard:
    """
    Global platform leaderboard across all medical imaging modalities.

    Attributes
    ----------
    rows : List[LeaderboardRow]
        Ranked entries.
    overall_platform_grade : str
        Platform-wide grade: A+ / A / B / C / D / F.
    top_performer : str
        Highest-ranked model entry description.
    coverage : Dict[str, int]
        Number of models per performance tier.
    """

    rows: List[LeaderboardRow]
    overall_platform_grade: str
    top_performer: str
    coverage: Dict[str, int]

    def to_markdown_table(self) -> str:
        header = (
            "| Rank | Tier | Modality | Model | Architecture | Score | F1 | Acc | ROC-AUC | Latency (ms) | RPS |\n"
            "|------|------|----------|-------|-------------|-------|-----|-----|---------|-------------|-----|\n"
        )
        rows_md = "\n".join(r.to_markdown_row() for r in self.rows)
        return header + rows_md

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall_platform_grade": self.overall_platform_grade,
            "top_performer": self.top_performer,
            "coverage": self.coverage,
            "rows": [r.to_dict() for r in self.rows],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class LeaderboardBuilder:
    """
    Constructs Leaderboard from ComparisonSummary.

    Platform-Wide Grade Rubric
    --------------------------
    Average Composite Score >= 0.90 → A+
    Average Composite Score >= 0.85 → A
    Average Composite Score >= 0.80 → B+
    Average Composite Score >= 0.75 → B
    Average Composite Score >= 0.70 → C
    else                            → D
    """

    GRADE_RUBRIC = [
        (0.90, "A+"),
        (0.85, "A"),
        (0.80, "B+"),
        (0.75, "B"),
        (0.70, "C"),
        (0.00, "D"),
    ]

    def build(self, summary: ComparisonSummary) -> Leaderboard:
        """Build Leaderboard from ComparisonSummary."""
        if not summary.entries:
            raise LeaderboardError("Cannot build leaderboard from empty comparison summary.")

        rows: List[LeaderboardRow] = []
        tier_coverage: Dict[str, int] = {t.name: 0 for t in LEADERBOARD_TIERS}

        for entry in summary.entries:
            tier = _get_tier(entry.composite_score)
            tier_coverage[tier.name] = tier_coverage.get(tier.name, 0) + 1

            row = LeaderboardRow(
                rank=entry.rank,
                tier=tier.name,
                tier_icon=tier.icon,
                modality=entry.modality,
                model_name=entry.model_name,
                architecture=entry.architecture,
                composite_score=entry.composite_score,
                f1_score=entry.f1_score,
                accuracy=entry.accuracy,
                roc_auc=entry.roc_auc,
                inference_latency_ms=entry.inference_latency_ms,
                throughput_rps=entry.throughput_rps,
                efficiency_ratio=entry.efficiency_ratio,
            )
            rows.append(row)

        # Platform grade
        avg_score = sum(r.composite_score for r in rows) / len(rows)
        grade = "D"
        for threshold, g in self.GRADE_RUBRIC:
            if avg_score >= threshold:
                grade = g
                break

        top = rows[0] if rows else None
        top_str = f"{top.modality} ({top.model_name}) — Score {top.composite_score:.4f}" if top else "N/A"

        logger.info("[Leaderboard] Platform Grade=%s AverageScore=%.4f", grade, avg_score)

        return Leaderboard(
            rows=rows,
            overall_platform_grade=grade,
            top_performer=top_str,
            coverage=tier_coverage,
        )


__all__ = ["Leaderboard", "LeaderboardBuilder", "LeaderboardRow", "LeaderboardTier"]
