"""
calibration/v2_phase7/sanity_analyzer.py
=========================================
Heuristic analysis of Grad-CAM activation patterns.

For each generated heatmap, classifies the dominant attention region:
  - central_focus:      Activation concentrated in central image region
  - border_artifact:    Disproportionate activation along image borders
  - diffuse:            Activation spread uniformly — no focused region
  - background_artifact: Strong activation outside expected anatomical region

This is qualitative/diagnostic analysis only. No quantitative claim of
"explanation accuracy" is made without an established localization ground truth.

IMPORTANT:
  - Heatmaps are NOT proof that a model understands pathology.
  - "central_focus" does NOT guarantee anatomically correct attention.
  - Results are labeled "attention_observation" not "clinical reasoning".

Output columns in gradcam_analysis.csv:
  modality, image_id, true_label, predicted_label, prediction_score,
  correct, category, attention_observation, artifact_suspected,
  border_ratio, center_ratio, diffuseness, overlay_path
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Spatial region definitions (as fraction of image size)
BORDER_BAND = 0.12  # Outer 12% of pixels constitute the border band
CENTER_BAND = 0.40  # Inner 40% constitutes the center region


def _load_heatmap_from_overlay(overlay_path: str) -> Optional[np.ndarray]:
    """
    Loads a saved Grad-CAM overlay and extracts the red channel
    as a proxy for heatmap intensity (JET colormap: red = high activation).

    Returns:
        Normalized heatmap array (H, W) in [0, 1], or None on failure
    """
    try:
        img = Image.open(overlay_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)
        # Red channel proxy for JET colormap activation intensity
        red = arr[:, :, 0] / 255.0
        return red
    except Exception as e:
        logger.warning(f"Failed to load overlay {overlay_path}: {e}")
        return None


def compute_spatial_metrics(
    heatmap: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Computes spatial activation ratios for border/center/diffuseness.

    Args:
        heatmap: 2D float array (H, W) in [0, 1]

    Returns:
        (border_ratio, center_ratio, diffuseness)
        border_ratio:  Mean activation in border band / global mean
        center_ratio:  Mean activation in center region / global mean
        diffuseness:   std / (mean + eps) — higher = more diffuse
    """
    h, w = heatmap.shape
    global_mean = float(heatmap.mean()) + 1e-8

    # Border mask: outer BORDER_BAND fraction of pixels
    bh = max(1, int(h * BORDER_BAND))
    bw = max(1, int(w * BORDER_BAND))
    border_mask = np.zeros((h, w), dtype=bool)
    border_mask[:bh, :] = True
    border_mask[-bh:, :] = True
    border_mask[:, :bw] = True
    border_mask[:, -bw:] = True
    border_mean = float(heatmap[border_mask].mean()) if border_mask.any() else 0.0
    border_ratio = border_mean / global_mean

    # Center mask: inner CENTER_BAND fraction
    ch_start = int(h * (0.5 - CENTER_BAND / 2))
    ch_end = int(h * (0.5 + CENTER_BAND / 2))
    cw_start = int(w * (0.5 - CENTER_BAND / 2))
    cw_end = int(w * (0.5 + CENTER_BAND / 2))
    center_region = heatmap[ch_start:ch_end, cw_start:cw_end]
    center_mean = float(center_region.mean()) if center_region.size > 0 else 0.0
    center_ratio = center_mean / global_mean

    # Diffuseness: coefficient of variation
    diffuseness = float(heatmap.std()) / global_mean

    return border_ratio, center_ratio, diffuseness


def classify_attention(
    border_ratio: float,
    center_ratio: float,
    diffuseness: float,
) -> Tuple[str, bool]:
    """
    Classifies dominant attention pattern and flags suspicious artifacts.

    Classification rules (heuristic, not clinically validated):
      border_artifact:      border_ratio > 1.4
      diffuse:              diffuseness < 0.25  (very uniform, no focus)
      central_focus:        center_ratio > 1.15 AND border_ratio < 1.2
      background_artifact:  border_ratio > 1.1 AND center_ratio < 0.9

    Args:
        border_ratio:  Border region mean / global mean
        center_ratio:  Center region mean / global mean
        diffuseness:   Coefficient of variation of heatmap

    Returns:
        (attention_observation, artifact_suspected)
    """
    artifact_suspected = False

    if border_ratio > 1.4:
        attention_observation = "border_artifact"
        artifact_suspected = True
    elif diffuseness < 0.25:
        attention_observation = "diffuse"
        artifact_suspected = False
    elif center_ratio > 1.15 and border_ratio < 1.2:
        attention_observation = "central_focus"
        artifact_suspected = False
    elif border_ratio > 1.1 and center_ratio < 0.9:
        attention_observation = "background_artifact"
        artifact_suspected = True
    else:
        attention_observation = "mixed"
        artifact_suspected = False

    return attention_observation, artifact_suspected


class GradCAMSanityAnalyzer:
    """
    Performs heuristic sanity analysis on Grad-CAM overlay images.
    """

    def analyze(
        self, gradcam_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes spatial activation patterns for all generated Grad-CAM records.

        Args:
            gradcam_records: Output from GradCAMEngine.process_sample_batch()

        Returns:
            Enriched list with attention_observation and artifact_suspected fields
        """
        enriched = []
        for rec in gradcam_records:
            overlay_path = rec.get("overlay_path")
            enriched_rec = dict(rec)

            if not overlay_path or not rec.get("gradcam_generated", False):
                enriched_rec.update({
                    "attention_observation": "not_generated",
                    "artifact_suspected": False,
                    "border_ratio": None,
                    "center_ratio": None,
                    "diffuseness": None,
                })
                enriched.append(enriched_rec)
                continue

            heatmap = _load_heatmap_from_overlay(overlay_path)
            if heatmap is None:
                enriched_rec.update({
                    "attention_observation": "load_error",
                    "artifact_suspected": False,
                    "border_ratio": None,
                    "center_ratio": None,
                    "diffuseness": None,
                })
                enriched.append(enriched_rec)
                continue

            border_ratio, center_ratio, diffuseness = compute_spatial_metrics(heatmap)
            attention_obs, artifact_flag = classify_attention(
                border_ratio, center_ratio, diffuseness
            )

            enriched_rec.update({
                "attention_observation": attention_obs,
                "artifact_suspected": artifact_flag,
                "border_ratio": round(border_ratio, 4),
                "center_ratio": round(center_ratio, 4),
                "diffuseness": round(diffuseness, 4),
            })
            enriched.append(enriched_rec)

        n_artifact = sum(1 for r in enriched if r.get("artifact_suspected", False))
        logger.info(
            f"Sanity analysis complete: {len(enriched)} records, "
            f"{n_artifact} with suspected artifacts"
        )
        return enriched

    def save(
        self, records: List[Dict[str, Any]], path: Path | str
    ) -> None:
        """Saves analysis results to gradcam_analysis.csv."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not records:
            return
        fieldnames = [
            "modality", "image_id", "image_path", "true_label",
            "predicted_label", "prediction_score", "correct", "category",
            "attention_observation", "artifact_suspected",
            "border_ratio", "center_ratio", "diffuseness", "overlay_path",
        ]
        # Ensure all fields exist in records
        sanitized = []
        for r in records:
            row = {k: r.get(k) for k in fieldnames}
            sanitized.append(row)

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(sanitized)
        logger.info(f"Grad-CAM sanity analysis saved to {path}")
