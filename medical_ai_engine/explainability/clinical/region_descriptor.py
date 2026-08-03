"""
region_descriptor.py — Module 15.5: Clinical Explanation Engine
Medical AI Platform — Converts classified finding dicts into spatial natural-language descriptions.

Responsibility
--------------
Translates each classified finding from FindingClassifier into a structured
English phrase describing location, size, and activation intensity.
Does NOT generate full clinical sentences — that is ClinicalTextGenerator's job.
"""

from __future__ import annotations

import logging
import time
from typing import List

from .clinical_configuration import ClinicalConfiguration
from .clinical_context import ClinicalContext
from .clinical_exceptions import RegionDescriptionError

logger = logging.getLogger(__name__)

# ── Activation intensity vocabulary ──────────────────────────────────────────
_INTENSITY_MAP = [
    (0.80, "intense"),
    (0.60, "prominent"),
    (0.40, "moderate"),
    (0.20, "mild"),
    (0.0,  "minimal"),
]

# ── Quadrant labels for spatial position ─────────────────────────────────────
_QUADRANT_MAP = {
    (0, 0): "upper-left",
    (0, 1): "upper-right",
    (1, 0): "lower-left",
    (1, 1): "lower-right",
}


def _intensity_label(peak: float) -> str:
    for threshold, label in _INTENSITY_MAP:
        if peak >= threshold:
            return label
    return "minimal"


def _quadrant_label(cx: float, cy: float) -> str:
    row = 0 if cy < 0.5 else 1
    col = 0 if cx < 0.5 else 1
    return _QUADRANT_MAP[(row, col)]


class RegionDescriptor:
    """
    Generates short natural-language spatial descriptions for each classified finding.

    Output format (standard verbosity):
        "{intensity} activation in the {anatomy} ({region_size}, {quadrant})"

    Examples:
        "intense activation in the frontal lobe (extensive, upper-left)"
        "moderate activation in the lung field (small, lower-right)"
    """

    def __init__(self, config: ClinicalConfiguration) -> None:
        self._config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def describe(self, ctx: ClinicalContext) -> List[str]:
        """
        Generate region descriptions from *ctx.classified_findings* and populate
        *ctx.region_descriptions*. Returns the list of description strings.
        """
        t0 = time.perf_counter()
        try:
            descs = self._generate(ctx)
            ctx.region_descriptions = descs
            ctx.record_stage_time(
                "region_description",
                (time.perf_counter() - t0) * 1000,
            )
            return descs
        except Exception as exc:
            msg = f"RegionDescriptor failed: {exc}"
            ctx.add_error(msg)
            logger.error(msg)
            raise RegionDescriptionError(msg) from exc

    # ── Internal ──────────────────────────────────────────────────────────────

    def _generate(self, ctx: ClinicalContext) -> List[str]:
        findings = ctx.classified_findings or []
        regions  = ctx.important_regions   or []
        descs: List[str] = []

        for finding in findings:
            anatomy    = finding.get("anatomy", "target region")
            peak       = float(finding.get("peak_score", 0.0))
            area       = float(finding.get("area_fraction", 0.0))
            region_idx = finding.get("region_index", -1)
            intensity  = _intensity_label(peak)
            size_label = self._config.classify_region_size(area)

            # Derive quadrant from region centroid when available
            if 0 <= region_idx < len(regions):
                r = regions[region_idx]
                cx = float(r.get("centroid_x", 0.5))
                cy = float(r.get("centroid_y", 0.5))
                quad = _quadrant_label(cx, cy)
            else:
                quad = "central"

            if self._config.verbosity == "brief":
                desc = f"{intensity} activation in the {anatomy}"
            elif self._config.verbosity == "detailed":
                desc = (
                    f"{intensity} activation (peak={peak:.2f}) in the {anatomy} "
                    f"covering a {size_label} area in the {quad} quadrant "
                    f"(area fraction={area:.1%})"
                )
            else:  # standard
                desc = (
                    f"{intensity} activation in the {anatomy} "
                    f"({size_label}, {quad})"
                )
            descs.append(desc)

        if not descs:
            descs.append("no significant focal activation detected")

        return descs


__all__ = ["RegionDescriptor"]
