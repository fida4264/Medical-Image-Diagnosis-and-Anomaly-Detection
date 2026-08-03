"""
eda/runner.py
==============
EDA pipeline orchestrator for Module 5.

Coordinates all analyzers, chart generation, and report writing for the
Medical Dataset Exploratory Data Analysis.

Pipeline Steps
--------------
[1/8]  Config         — Load configs/eda.yaml
[2/8]  Load Data      — Read master_metadata.csv (Module 2 output)
[3/8]  Dataset Stats  — DatasetAnalyzer (overall composition)
[4/8]  Class Balance  — ClassAnalyzer (imbalance, Gini, rare classes)
[5/8]  Modality Stats — ModalityAnalyzer (per-modality breakdown)
[6/8]  Image Props    — ImagePropertyAnalyzer (resolution, pixel stats)
[7/8]  Quality        — QualityAnalyzer (flags, Dataset Quality Score)
[8/8]  Preprocessing  — PreprocessingAnalyzer + AugmentationAnalyzer
       Visualizations — Charts, heatmaps, dashboards
       Reports        — JSON, Markdown, CSV

Classes
-------
EDARunner
    Orchestrates the end-to-end EDA pipeline.

Author: Medical AI System — Module 5
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml

from eda.config import (
    EDA_OUTPUT_DIR,
    EDA_VISUALIZATIONS_DIR,
    EDA_CONFIG_PATH,
    EDA_STATISTICS_JSON,
    EDA_SUMMARY_MD,
    EDA_TABLES_CSV,
    DATASET_QUALITY_SCORE_JSON,
    VISUALIZATION_INDEX_MD,
    MASTER_METADATA_CSV,
    COL_MODALITY,
    COL_NORMALIZED_CLASS,
    COL_WIDTH,
    COL_HEIGHT,
)
from eda.analyzers import (
    DatasetAnalyzer,
    ClassAnalyzer,
    ModalityAnalyzer,
    ImagePropertyAnalyzer,
    QualityAnalyzer,
    PreprocessingAnalyzer,
    AugmentationAnalyzer,
)
from eda.visualizations.charts import ChartGenerator
from eda.visualizations.heatmaps import HeatmapGenerator
from eda.visualizations.dashboard import DashboardGenerator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _step(n: int, total: int, msg: str) -> None:
    logger.info("[%d/%d] %s", n, total, msg)
    print(f"\n{'=' * 78}\n  [{n}/{total}] {msg}\n{'=' * 78}")


def _ok(msg: str, elapsed: float) -> None:
    print(f"  [OK] {msg} ({elapsed:.2f}s)")


# ---------------------------------------------------------------------------
# EDARunner
# ---------------------------------------------------------------------------


class EDARunner:
    """
    Orchestrates the full Module 5 EDA pipeline.

    Parameters
    ----------
    config_path : Path, optional
        Path to configs/eda.yaml. Defaults to config value.
    output_dir : Path, optional
        Root output directory for EDA artifacts. Defaults to outputs/eda.
    sample_mode : bool
        If True, analyzes a small sample per modality (useful for testing).
    sample_size : int
        Rows to sample per modality in sample mode.
    modality_filter : str, optional
        If provided, only analyze this modality.
    full_pixel_analysis : bool
        If True, override config pixel_analysis_sample → analyze all images.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        sample_mode: bool = False,
        sample_size: int = 500,
        modality_filter: Optional[str] = None,
        full_pixel_analysis: bool = False,
    ) -> None:
        self._config_path = Path(config_path) if config_path else EDA_CONFIG_PATH
        self._output_dir  = Path(output_dir) if output_dir else EDA_OUTPUT_DIR
        self._vis_dir     = self._output_dir / "visualizations"
        self._sample_mode = sample_mode
        self._sample_size = sample_size
        self._modality_filter = modality_filter
        self._full_pixel = full_pixel_analysis

        self._config: Dict[str, Any] = {}
        self._df: Optional[pd.DataFrame] = None

        # Collected stats from all analyzers
        self._stats: Dict[str, Any] = {}
        # All saved chart paths
        self._chart_paths: List[Path] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Execute the full EDA pipeline.

        Returns
        -------
        dict
            Aggregated statistics from all analyzers.
        """
        total_steps = 8
        pipeline_start = time.perf_counter()

        # ------------------------------------------------------------------
        # Step 1 — Load config
        # ------------------------------------------------------------------
        _step(1, total_steps, "Loading configs/eda.yaml ...")
        t0 = time.perf_counter()
        self._config = self._load_config()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._vis_dir.mkdir(parents=True, exist_ok=True)
        _ok("Config loaded", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Step 2 — Load master_metadata.csv
        # ------------------------------------------------------------------
        _step(2, total_steps, "Loading master_metadata.csv ...")
        t0 = time.perf_counter()
        self._df = self._load_master_metadata()
        n_rows = len(self._df)
        n_mods = self._df[COL_MODALITY].nunique() if COL_MODALITY in self._df.columns else 0
        _ok(f"{n_rows:,} images | {n_mods} modalities", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Step 3 — Dataset composition analysis
        # ------------------------------------------------------------------
        _step(3, total_steps, "Analysing dataset composition ...")
        t0 = time.perf_counter()
        da = DatasetAnalyzer()
        da._df = self._df  # inject already-loaded df
        da._registry = {}  # registry optionally loaded inside analyze()
        dataset_stats = da.analyze()
        self._stats["dataset"] = dataset_stats
        _ok(
            f"{dataset_stats['total_images']:,} images | "
            f"{dataset_stats['total_modalities']} modalities | "
            f"{dataset_stats['total_disease_classes']} classes",
            time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Step 4 — Class balance analysis
        # ------------------------------------------------------------------
        _step(4, total_steps, "Analysing class balance ...")
        t0 = time.perf_counter()
        analysis_cfg = self._config.get("analysis", {})
        class_cfg = analysis_cfg.get("class_balance", {})
        ca = ClassAnalyzer(
            df=self._df,
            rare_class_threshold=int(class_cfg.get("rare_class_threshold", 100)),
            minority_ratio_threshold=float(class_cfg.get("minority_ratio_threshold", 0.1)),
        )
        class_stats = ca.analyze()
        self._stats["class_balance"] = class_stats
        _ok(
            f"{class_stats.get('total_classes', 0)} classes | "
            f"imbalance={class_stats.get('imbalance_ratio', 0):.1f}× | "
            f"{class_stats.get('balance_label', 'N/A')}",
            time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Step 5 — Modality analysis
        # ------------------------------------------------------------------
        _step(5, total_steps, "Analysing modalities ...")
        t0 = time.perf_counter()
        ma = ModalityAnalyzer(df=self._df)
        modality_stats = ma.analyze()
        self._stats["modality"] = modality_stats
        _ok(f"{modality_stats.get('total_modalities', 0)} modalities", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Step 6 — Image property analysis
        # ------------------------------------------------------------------
        _step(6, total_steps, "Analysing image properties ...")
        t0 = time.perf_counter()
        img_cfg = analysis_cfg.get("image_properties", {})
        general_cfg = self._config.get("general", {})
        pixel_sample = (
            len(self._df) if self._full_pixel
            else int(general_cfg.get("pixel_analysis_sample", 2000))
        )
        compute_px = bool(img_cfg.get("compute_pixel_stats", True))
        ipa = ImagePropertyAnalyzer(
            df=self._df,
            pixel_sample_size=pixel_sample,
            compute_pixel_stats=compute_px,
            random_seed=int(general_cfg.get("random_seed", 42)),
        )
        image_stats = ipa.analyze()
        self._stats["image_properties"] = image_stats
        _ok("Image property analysis complete", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Step 7 — Quality analysis
        # ------------------------------------------------------------------
        _step(7, total_steps, "Analysing dataset quality ...")
        t0 = time.perf_counter()
        class_gini = float(class_stats.get("gini_coefficient", 0.0))
        qa = QualityAnalyzer(master_df=self._df, class_gini=class_gini)
        qa.load()
        quality_stats = qa.analyze()
        self._stats["quality"] = quality_stats
        q_score = quality_stats.get("quality_score", {})
        _ok(
            f"Quality score: {q_score.get('total_score', 0):.1f}/100 "
            f"({q_score.get('grade', 'N/A')})",
            time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Step 8 — Preprocessing + Augmentation analysis
        # ------------------------------------------------------------------
        _step(8, total_steps, "Analysing preprocessing & augmentation ...")
        t0 = time.perf_counter()
        ppa = PreprocessingAnalyzer()
        ppa.load()
        prep_stats = ppa.analyze()
        self._stats["preprocessing"] = prep_stats

        aug_analyzer = AugmentationAnalyzer(master_df=self._df)
        aug_analyzer.load()
        aug_stats = aug_analyzer.analyze()
        self._stats["augmentation"] = aug_stats

        prep_ok = prep_stats.get("available", False)
        aug_ok  = aug_stats.get("available", False)
        _ok(
            f"Preprocessing: {'available' if prep_ok else 'not found'} | "
            f"Augmentation: {'available' if aug_ok else 'not found'}",
            time.perf_counter() - t0,
        )

        # ------------------------------------------------------------------
        # Visualizations
        # ------------------------------------------------------------------
        print(f"\n{'=' * 78}\n  Generating visualisations ...\n{'=' * 78}")
        t0 = time.perf_counter()
        self._generate_charts(dataset_stats, class_stats, modality_stats, quality_stats, aug_stats)
        _ok(f"{len(self._chart_paths)} charts saved -> {self._vis_dir}", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Reports
        # ------------------------------------------------------------------
        print(f"\n{'=' * 78}\n  Writing reports ...\n{'=' * 78}")
        t0 = time.perf_counter()
        self._write_reports(quality_stats)
        _ok("Reports written", time.perf_counter() - t0)

        # ------------------------------------------------------------------
        # Final summary
        # ------------------------------------------------------------------
        total_elapsed = time.perf_counter() - pipeline_start
        print(f"\n{'=' * 78}")
        print("  MODULE 5 — EDA COMPLETE")
        print(f"{'=' * 78}")
        print(f"  Output directory    : {self._output_dir}")
        print(f"  Charts generated    : {len(self._chart_paths)}")
        print(f"  Total images        : {dataset_stats.get('total_images', 0):,}")
        print(f"  Modalities          : {dataset_stats.get('total_modalities', 0)}")
        print(f"  Disease classes     : {dataset_stats.get('total_disease_classes', 0)}")
        print(f"  Dataset quality     : {q_score.get('total_score', 0):.1f}/100  {q_score.get('grade', '')}")
        print(f"  Total time          : {total_elapsed:.1f}s")
        print(f"{'=' * 78}\n")
        logger.info("Module 5 EDA complete in %.1fs.", total_elapsed)

        return self._stats

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> Dict[str, Any]:
        """Load eda.yaml; return {} if not found."""
        if self._config_path.exists():
            with open(self._config_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            logger.info("Config loaded: %s", self._config_path)
            return cfg
        logger.warning("eda.yaml not found at %s — using defaults.", self._config_path)
        return {}

    def _load_master_metadata(self) -> pd.DataFrame:
        """Load master_metadata.csv (Module 2 output)."""
        if not MASTER_METADATA_CSV.exists():
            raise FileNotFoundError(
                f"master_metadata.csv not found: {MASTER_METADATA_CSV}\n"
                "Run Module 2 (run_metadata_manager.py) first."
            )
        df = pd.read_csv(MASTER_METADATA_CSV, encoding="utf-8", low_memory=False)
        logger.info("Loaded master_metadata.csv: %d rows", len(df))

        # Apply modality filter
        if self._modality_filter and COL_MODALITY in df.columns:
            df = df[df[COL_MODALITY] == self._modality_filter]
            logger.info("Filtered to modality '%s': %d rows", self._modality_filter, len(df))

        # Apply sample mode
        if self._sample_mode and COL_MODALITY in df.columns:
            df = (
                df.groupby(COL_MODALITY, group_keys=False)
                .apply(
                    lambda g: g.sample(min(self._sample_size, len(g)), random_state=42),
                    include_groups=False,
                )
                .reset_index(drop=True)
            )
            logger.info("SAMPLE MODE: %d rows (%d per modality)", len(df), self._sample_size)

        return df

    def _generate_charts(
        self,
        dataset_stats: Dict[str, Any],
        class_stats: Dict[str, Any],
        modality_stats: Dict[str, Any],
        quality_stats: Dict[str, Any],
        aug_stats: Dict[str, Any],
    ) -> None:
        """Generate all EDA charts and dashboards."""
        charts = ChartGenerator(output_dir=self._vis_dir)
        heatmaps = HeatmapGenerator(output_dir=self._vis_dir)
        dashboards = DashboardGenerator(output_dir=self._vis_dir)
        vis_cfg = self._config.get("visualizations", {})

        def _enabled(key: str) -> bool:
            return bool(vis_cfg.get(key, True))

        # -------------------------------------------------------------------
        # Dataset composition charts
        # -------------------------------------------------------------------
        if _enabled("modality_pie"):
            mod_counts = dataset_stats.get("images_per_modality", {})
            if mod_counts:
                self._chart_paths.append(
                    charts.pie(mod_counts, "Images by Modality", "modality_pie")
                )

        if _enabled("dataset_source_bar"):
            src_counts = dataset_stats.get("images_per_source", {})
            if src_counts:
                self._chart_paths.append(
                    charts.bar_h(src_counts, "Images per Dataset Source", "dataset_source_bar")
                )

        if _enabled("train_test_split"):
            split_counts = dataset_stats.get("split_distribution", {})
            if split_counts:
                self._chart_paths.append(
                    charts.pie(split_counts, "Train / Val / Test Split", "split_distribution_pie")
                )

        # -------------------------------------------------------------------
        # Class balance charts
        # -------------------------------------------------------------------
        if _enabled("class_bar"):
            class_counts = class_stats.get("class_counts", {})
            if class_counts:
                top30 = dict(
                    sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:30]
                )
                self._chart_paths.append(
                    charts.bar_h(top30, "Top 30 Disease Classes by Image Count", "class_bar")
                )

        if _enabled("class_treemap"):
            class_counts = class_stats.get("class_counts", {})
            if class_counts:
                self._chart_paths.append(
                    charts.treemap(class_counts, "Disease Class Distribution Treemap", "class_treemap")
                )

        if _enabled("class_violin"):
            per_mod_cls = class_stats.get("per_modality", {})
            violin_data: Dict[str, Any] = {}
            for mod, data in per_mod_cls.items():
                counts_list = list(data.get("class_counts", {}).values())
                if counts_list:
                    violin_data[mod] = counts_list
            if violin_data:
                self._chart_paths.append(
                    charts.violin(
                        violin_data,
                        "Class Count Distribution per Modality",
                        "class_violin_by_modality",
                        xlabel="Modality",
                        ylabel="Images per Class",
                    )
                )

        # -------------------------------------------------------------------
        # Image property charts
        # -------------------------------------------------------------------
        img_props = self._stats.get("image_properties", {})
        physical = img_props.get("physical", {})

        if _enabled("resolution_scatter") and COL_WIDTH in self._df.columns and COL_HEIGHT in self._df.columns:
            sample_df = self._df.sample(min(5000, len(self._df)), random_state=42)
            valid = sample_df[(sample_df[COL_WIDTH] > 0) & (sample_df[COL_HEIGHT] > 0)]
            if not valid.empty:
                labels_list = None
                if COL_MODALITY in valid.columns:
                    labels_list = valid[COL_MODALITY].tolist()
                self._chart_paths.append(
                    charts.scatter(
                        valid[COL_WIDTH].tolist(),
                        valid[COL_HEIGHT].tolist(),
                        "Image Resolution Scatter (Width vs Height)",
                        "resolution_scatter",
                        xlabel="Width (px)",
                        ylabel="Height (px)",
                        labels=labels_list,
                        s=8,
                    )
                )

        if _enabled("resolution_histogram") and COL_WIDTH in self._df.columns:
            widths = self._df[self._df[COL_WIDTH] > 0][COL_WIDTH].tolist()
            if widths:
                self._chart_paths.append(
                    charts.histogram(widths, "Image Width Distribution", "resolution_width_histogram",
                                     xlabel="Width (pixels)")
                )

        if _enabled("aspect_ratio_histogram") and COL_WIDTH in self._df.columns and COL_HEIGHT in self._df.columns:
            valid = self._df[(self._df[COL_WIDTH] > 0) & (self._df[COL_HEIGHT] > 0)]
            if not valid.empty:
                ar = (valid[COL_WIDTH] / valid[COL_HEIGHT]).tolist()
                self._chart_paths.append(
                    charts.histogram(ar, "Aspect Ratio Distribution", "aspect_ratio_histogram",
                                     xlabel="Aspect Ratio (W/H)", bins=40)
                )

        if _enabled("file_size_histogram"):
            from eda.config import COL_FILE_SIZE
            if COL_FILE_SIZE in self._df.columns:
                fs_kb = (self._df[COL_FILE_SIZE].dropna() / 1024).tolist()
                if fs_kb:
                    self._chart_paths.append(
                        charts.histogram(fs_kb, "File Size Distribution", "file_size_histogram",
                                         xlabel="File Size (KB)", bins=50)
                    )

        # Resolution density heatmap
        if COL_WIDTH in self._df.columns and COL_HEIGHT in self._df.columns:
            valid = self._df[(self._df[COL_WIDTH] > 0) & (self._df[COL_HEIGHT] > 0)]
            if not valid.empty:
                self._chart_paths.append(
                    heatmaps.resolution_density_heatmap(
                        valid[COL_WIDTH].tolist(),
                        valid[COL_HEIGHT].tolist(),
                        "Resolution Density (Width vs Height)",
                        "resolution_density_heatmap",
                    )
                )

        # Pixel intensity histogram (per modality)
        if _enabled("pixel_intensity_histogram"):
            pixel_stats = img_props.get("pixel", {})
            for mod, p_data in pixel_stats.items():
                hist_data = p_data.get("intensity_histogram", {})
                counts = hist_data.get("counts", [])
                bins   = hist_data.get("bins", [])
                if counts and bins:
                    # Convert histogram counts back to a sample series for our histogram
                    # Use midpoints as proxy values
                    mids = [(bins[i] + bins[i + 1]) / 2 for i in range(len(counts))]
                    expanded = []
                    for mid, cnt in zip(mids, counts):
                        expanded.extend([mid] * cnt)
                    if expanded:
                        self._chart_paths.append(
                            charts.histogram(
                                expanded,
                                f"Pixel Intensity Distribution — {mod}",
                                f"pixel_intensity_{mod}",
                                xlabel="Mean Pixel Intensity",
                                bins=50,
                            )
                        )

        # -------------------------------------------------------------------
        # Heatmaps
        # -------------------------------------------------------------------
        if _enabled("class_modality_heatmap") and COL_NORMALIZED_CLASS in self._df.columns and COL_MODALITY in self._df.columns:
            self._chart_paths.append(
                heatmaps.class_modality_heatmap(
                    self._df,
                    class_col=COL_NORMALIZED_CLASS,
                    modality_col=COL_MODALITY,
                    title="Disease Class × Modality Image Count Heatmap",
                    filename="class_modality_heatmap",
                )
            )

        if _enabled("correlation_heatmap"):
            numeric_cols = [c for c in [COL_WIDTH, COL_HEIGHT, "file_size_bytes", "channels"]
                            if c in self._df.columns]
            if len(numeric_cols) >= 2:
                self._chart_paths.append(
                    heatmaps.correlation_heatmap(
                        self._df,
                        title="Numeric Feature Correlation Heatmap",
                        filename="numeric_correlation_heatmap",
                        columns=numeric_cols,
                    )
                )

        # -------------------------------------------------------------------
        # Dashboards
        # -------------------------------------------------------------------
        if _enabled("dataset_overview_dashboard"):
            self._chart_paths.append(
                dashboards.dataset_overview(
                    dataset_stats=dataset_stats,
                    class_stats=class_stats,
                    modality_stats=modality_stats,
                    filename="dashboard_dataset_overview",
                )
            )

        if _enabled("quality_dashboard"):
            self._chart_paths.append(
                dashboards.quality_dashboard(
                    quality_stats=quality_stats,
                    filename="dashboard_quality",
                )
            )

        if _enabled("augmentation_dashboard"):
            self._chart_paths.append(
                dashboards.augmentation_dashboard(
                    aug_stats=aug_stats,
                    filename="dashboard_augmentation",
                )
            )

    def _write_reports(self, quality_stats: Dict[str, Any]) -> None:
        """Write JSON, Markdown, CSV, and quality score reports."""
        reports_cfg = self._config.get("reports", {})

        # --- JSON: eda_statistics.json ---
        if reports_cfg.get("generate_json", True):
            json_path = self._output_dir / EDA_STATISTICS_JSON
            with open(json_path, "w", encoding="utf-8") as fh:
                json.dump(self._stats, fh, indent=2, default=str)
            logger.info("Written: %s", json_path)

        # --- JSON: dataset_quality_score.json ---
        q_score = quality_stats.get("quality_score", {})
        qs_path = self._output_dir / DATASET_QUALITY_SCORE_JSON
        with open(qs_path, "w", encoding="utf-8") as fh:
            json.dump(q_score, fh, indent=2, default=str)
        logger.info("Written: %s", qs_path)

        # --- Markdown: eda_summary.md ---
        if reports_cfg.get("generate_markdown", True):
            md_path = self._output_dir / EDA_SUMMARY_MD
            self._write_markdown_summary(md_path)
            logger.info("Written: %s", md_path)

        # --- CSV: eda_tables.csv (class counts) ---
        if reports_cfg.get("generate_csv", True):
            class_stats = self._stats.get("class_balance", {})
            class_counts = class_stats.get("class_counts", {})
            if class_counts:
                import pandas as pd
                csv_path = self._output_dir / EDA_TABLES_CSV
                pd.DataFrame(
                    list(class_counts.items()),
                    columns=["disease_class", "image_count"],
                ).to_csv(csv_path, index=False, encoding="utf-8")
                logger.info("Written: %s", csv_path)

        # --- Markdown: visualization_index.md ---
        vis_index_path = self._output_dir / VISUALIZATION_INDEX_MD
        self._write_visualization_index(vis_index_path)
        logger.info("Written: %s", vis_index_path)

    def _write_markdown_summary(self, path: Path) -> None:
        """Write a human-readable Markdown EDA summary report."""
        ds  = self._stats.get("dataset", {})
        cls = self._stats.get("class_balance", {})
        q   = self._stats.get("quality", {})
        qs  = q.get("quality_score", {})
        aug = self._stats.get("augmentation", {})
        pre = self._stats.get("preprocessing", {})

        lines = [
            "# Medical Dataset — EDA Summary Report",
            "",
            f"> Generated by Module 5 | Medical AI System",
            "",
            "---",
            "",
            "## 1. Dataset Overview",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Images | {ds.get('total_images', 0):,} |",
            f"| Modalities | {ds.get('total_modalities', 0)} |",
            f"| Disease Classes | {ds.get('total_disease_classes', 0)} |",
            f"| Dataset Sources | {ds.get('total_dataset_sources', 0)} |",
            "",
            "### Images per Modality",
            "",
        ]

        for mod, cnt in ds.get("images_per_modality", {}).items():
            display = ds.get("modality_display_names", {}).get(mod, mod)
            lines.append(f"- **{display}**: {cnt:,}")

        lines += [
            "",
            "---",
            "",
            "## 2. Class Balance",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Classes | {cls.get('total_classes', 0)} |",
            f"| Gini Coefficient | {cls.get('gini_coefficient', 0):.4f} |",
            f"| Imbalance Ratio | {cls.get('imbalance_ratio', 0):.1f}× |",
            f"| Balance Assessment | **{cls.get('balance_label', 'N/A')}** |",
            f"| Rare Classes (< 100 images) | {cls.get('rare_class_count', 0)} |",
            f"| Minority Classes (< 10% of max) | {cls.get('minority_class_count', 0)} |",
            "",
            "---",
            "",
            "## 3. Dataset Quality Score",
            "",
            f"### Overall: **{qs.get('total_score', 0):.1f} / 100 — {qs.get('grade', 'N/A')}**",
            "",
            f"| Component | Score | Max |",
            f"|-----------|-------|-----|",
            f"| Class Balance | {qs.get('class_balance_score', 0):.1f} | 25 |",
            f"| Image Quality | {qs.get('image_quality_score', 0):.1f} | 25 |",
            f"| Corruption | {qs.get('corruption_score', 0):.1f} | 25 |",
            f"| Resolution Consistency | {qs.get('resolution_consistency_score', 0):.1f} | 25 |",
            "",
        ]

        flag_rate = q.get("overall_flag_rate_pct", 0)
        lines += [
            "### Quality Flags",
            "",
            f"- Overall flag rate: **{flag_rate:.1f}%**",
            "",
        ]
        for flag, data in q.get("flag_breakdown", {}).items():
            lines.append(f"- `{flag}`: {data.get('count', 0):,} images ({data.get('pct', 0):.1f}%)")

        lines += [
            "",
            "---",
            "",
            "## 4. Preprocessing Summary",
            "",
        ]
        if pre.get("available"):
            lines += [
                f"- Total records: {pre.get('total_records', 0):,}",
                f"- Success rate: {pre.get('success_rate_pct', 0):.1f}%",
                f"- Mean processing time: {pre.get('processing_time', {}).get('mean_ms', 0):.1f} ms/image",
            ]
        else:
            lines.append("_Preprocessing data not found (run Module 3 first)._")

        lines += [
            "",
            "---",
            "",
            "## 5. Augmentation Summary",
            "",
        ]
        if aug.get("available"):
            cb = aug.get("class_balance_improvement", {})
            gini_before = cb.get("gini_before", 0)
            gini_after  = cb.get("gini_after", 0)
            lines += [
                f"- Total augmented records: {aug.get('total_augmented_records', 0):,}",
                f"- Success rate: {aug.get('success_rate_pct', 0):.1f}%",
                f"- Augmentation factor: {aug.get('augmentation_factor', 0):.1f}×",
                f"- Gini before augmentation: {gini_before:.4f}",
                f"- Gini after augmentation: {gini_after:.4f}",
                f"- Improvement: {cb.get('improvement', 0):.4f}",
            ]
        else:
            lines.append("_Augmentation data not found (run Module 4 first)._")

        lines += [
            "",
            "---",
            "",
            "## 6. Visualisations",
            "",
            f"Charts saved in: `{self._vis_dir.relative_to(self._vis_dir.parent.parent)}`",
            "",
            f"Total charts generated: **{len(self._chart_paths)}**",
            "",
            "---",
            "",
            "_Report generated by Medical AI System — Module 5: Exploratory Data Analysis_",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")

    def _write_visualization_index(self, path: Path) -> None:
        """Write a Markdown index of all saved visualization files."""
        lines = [
            "# EDA Visualization Index",
            "",
            f"Total charts: {len(self._chart_paths)}",
            "",
            "| # | File | Path |",
            "|---|------|------|",
        ]
        for i, p in enumerate(self._chart_paths, 1):
            lines.append(f"| {i} | {p.name} | `{p}` |")

        path.write_text("\n".join(lines), encoding="utf-8")

    @property
    def stats(self) -> Dict[str, Any]:
        """Return the aggregated statistics from all analyzers."""
        return self._stats

    @property
    def chart_paths(self) -> List[Path]:
        """Return the list of saved chart file paths."""
        return self._chart_paths
