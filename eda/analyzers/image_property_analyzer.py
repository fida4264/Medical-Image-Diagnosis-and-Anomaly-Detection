"""
eda/analyzers/image_property_analyzer.py
==========================================
Analyzes image physical properties: resolution, aspect ratio, channels,
file size, and pixel-level statistics (sampled).

For pixel-level statistics (mean intensity, std, histograms), a configurable
sample of images is used to keep runtime feasible on large datasets.

Classes
-------
ImagePropertyAnalyzer
    Computes image property statistics from master_metadata.csv.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from eda.config import (
    COL_MODALITY,
    COL_IMAGE_PATH,
    COL_WIDTH,
    COL_HEIGHT,
    COL_CHANNELS,
    COL_FILE_SIZE,
    COL_EXTENSION,
    GRAYSCALE_MODALITIES,
)

logger = logging.getLogger(__name__)

_HISTOGRAM_BINS = 50


def _safe_read(path: str) -> Optional[np.ndarray]:
    """Load an image via cv2; return None on failure."""
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img
    except Exception:
        return None


class ImagePropertyAnalyzer:
    """
    Analyzes image physical and pixel-level properties.

    Parameters
    ----------
    df : pd.DataFrame
        The master_metadata DataFrame.
    pixel_sample_size : int
        Max images to sample per modality for pixel-level stats.
    compute_pixel_stats : bool
        If False, skip pixel-level analysis entirely.
    random_seed : int
        Random seed for reproducible sampling.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pixel_sample_size: int = 2000,
        compute_pixel_stats: bool = True,
        random_seed: int = 42,
    ) -> None:
        self._df = df
        self._pixel_sample = pixel_sample_size
        self._compute_pixels = compute_pixel_stats
        self._seed = random_seed
        self._stats: Dict[str, Any] = {}

    def analyze(self) -> Dict[str, Any]:
        """
        Compute image property statistics.

        Returns
        -------
        dict
            Physical and pixel-level property statistics.
        """
        df = self._df
        logger.info("Running ImagePropertyAnalyzer ...")

        # --- Physical properties from metadata columns ---
        physical = self._compute_physical(df)

        # --- Pixel-level stats (sampled) ---
        pixel_stats: Dict[str, Any] = {}
        if self._compute_pixels and COL_IMAGE_PATH in df.columns:
            pixel_stats = self._compute_pixel_stats(df)

        self._stats = {
            "physical": physical,
            "pixel": pixel_stats,
        }

        logger.info("ImagePropertyAnalyzer complete.")
        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_physical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute resolution, aspect ratio, file size stats from metadata."""
        result: Dict[str, Any] = {}

        if COL_WIDTH in df.columns and COL_HEIGHT in df.columns:
            valid = df[(df[COL_WIDTH] > 0) & (df[COL_HEIGHT] > 0)].copy()
            valid["aspect_ratio"] = valid[COL_WIDTH] / valid[COL_HEIGHT]

            result["resolution"] = {
                "width": self._describe(valid[COL_WIDTH]),
                "height": self._describe(valid[COL_HEIGHT]),
                "aspect_ratio": self._describe(valid["aspect_ratio"]),
            }

            # Resolution buckets
            def _bucket(row: pd.Series) -> str:
                w, h = row[COL_WIDTH], row[COL_HEIGHT]
                mn = min(w, h)
                if mn < 64:     return "<64"
                if mn < 128:    return "64–127"
                if mn < 224:    return "128–223"
                if mn == 224:   return "224 (standard)"
                if mn < 512:    return "225–511"
                return "≥512"

            buckets = valid.apply(_bucket, axis=1).value_counts().to_dict()
            result["resolution_buckets"] = buckets

            # Per-modality resolution
            if COL_MODALITY in valid.columns:
                per_mod: Dict[str, Any] = {}
                for mod, grp in valid.groupby(COL_MODALITY):
                    per_mod[mod] = {
                        "width_mean": round(float(grp[COL_WIDTH].mean()), 1),
                        "height_mean": round(float(grp[COL_HEIGHT].mean()), 1),
                        "aspect_ratio_mean": round(float(grp["aspect_ratio"].mean()), 3),
                    }
                result["resolution_per_modality"] = per_mod

        # File size
        if COL_FILE_SIZE in df.columns:
            fs = df[COL_FILE_SIZE].dropna()
            result["file_size"] = self._describe(fs, unit="bytes")
            # Histogram bin edges for file size (KB)
            fs_kb = fs / 1024
            hist, edges = np.histogram(fs_kb, bins=_HISTOGRAM_BINS)
            result["file_size_histogram"] = {
                "bins_kb": [round(float(e), 2) for e in edges.tolist()],
                "counts": hist.tolist(),
            }

        # Channels
        if COL_CHANNELS in df.columns:
            result["channel_distribution"] = (
                df[COL_CHANNELS].value_counts().to_dict()
            )

        # Extension
        if COL_EXTENSION in df.columns:
            result["extension_distribution"] = (
                df[COL_EXTENSION].str.lower().value_counts().to_dict()
            )

        return result

    def _compute_pixel_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Sample images and compute mean intensity, std, and histogram.
        Runs per-modality to capture RGB vs grayscale differences.
        """
        if COL_MODALITY not in df.columns:
            return self._pixel_stats_for_group(df, "all")

        result: Dict[str, Any] = {}
        for mod, grp in df.groupby(COL_MODALITY):
            sample = grp.sample(
                min(self._pixel_sample, len(grp)),
                random_state=self._seed,
            )
            logger.info(
                "  Pixel analysis: %s — sampling %d/%d images",
                mod, len(sample), len(grp),
            )
            result[mod] = self._pixel_stats_for_group(sample, mod)
        return result

    def _pixel_stats_for_group(
        self,
        df_group: pd.DataFrame,
        mod_name: str,
    ) -> Dict[str, Any]:
        """Read sampled images and compute mean/std/histogram."""
        means: List[float] = []
        stds: List[float] = []
        channel_means: Dict[str, List[float]] = {"R": [], "G": [], "B": []}
        failed = 0
        is_gray = mod_name in GRAYSCALE_MODALITIES

        for _, row in df_group.iterrows():
            path = row.get(COL_IMAGE_PATH, "")
            img = _safe_read(str(path))
            if img is None:
                failed += 1
                continue

            if is_gray:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                means.append(float(gray.mean()))
                stds.append(float(gray.std()))
            else:
                b, g, r = cv2.split(img)
                m = float(img.mean())
                means.append(m)
                stds.append(float(img.std()))
                channel_means["R"].append(float(r.mean()))
                channel_means["G"].append(float(g.mean()))
                channel_means["B"].append(float(b.mean()))

        if not means:
            return {"error": "no readable images in sample", "failed": failed}

        # Aggregate histogram across sampled means
        hist, edges = np.histogram(means, bins=_HISTOGRAM_BINS, range=(0, 255))

        result: Dict[str, Any] = {
            "sample_size": len(means),
            "failed_reads": failed,
            "mean_intensity": round(float(np.mean(means)), 2),
            "std_intensity": round(float(np.std(means)), 2),
            "mean_of_stds": round(float(np.mean(stds)), 2),
            "intensity_histogram": {
                "bins": [round(float(e), 1) for e in edges.tolist()],
                "counts": hist.tolist(),
            },
        }

        if not is_gray and channel_means["R"]:
            result["channel_means"] = {
                "R": round(float(np.mean(channel_means["R"])), 2),
                "G": round(float(np.mean(channel_means["G"])), 2),
                "B": round(float(np.mean(channel_means["B"])), 2),
            }

        return result

    @staticmethod
    def _describe(series: pd.Series, unit: str = "") -> Dict[str, Any]:
        """Return descriptive statistics for a numeric Series."""
        clean = series.dropna()
        if clean.empty:
            return {}
        return {
            "min": round(float(clean.min()), 2),
            "max": round(float(clean.max()), 2),
            "mean": round(float(clean.mean()), 2),
            "median": round(float(clean.median()), 2),
            "std": round(float(clean.std()), 2),
            "q25": round(float(clean.quantile(0.25)), 2),
            "q75": round(float(clean.quantile(0.75)), 2),
            "unit": unit,
        }

    @property
    def stats(self) -> Dict[str, Any]:
        return self._stats
