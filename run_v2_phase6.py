"""
run_v2_phase6.py
================
Master CLI runner for Phase 6: Baseline Evaluation, Error Analysis & Validation.
Executes all validation, sanity audits, confidence breakdowns, and report builds.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from evaluation.v2_phase6.error_analyzer import ErrorAnalyzer
from evaluation.v2_phase6.gap_and_cv_analyzer import GapAndCVAnalyzer
from evaluation.v2_phase6.report_builder import Phase6ReportBuilder
from evaluation.v2_phase6.sanity_checker import HistopathologySanityChecker
from evaluation.v2_phase6.validator import BaselineValidator
from training.v2_baseline.config import ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_v2_phase6")


def main():
    start_time = time.time()
    logger.info("=================================================================")
    logger.info("   STARTING V2.0 PHASE 6: EVALUATION & ERROR ANALYSIS PIPELINE   ")
    logger.info("=================================================================")

    output_dir = ROOT / "outputs" / "v2_phase6"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate Baselines & Generate Sample Predictions
    logger.info("\n>>> Step 1: Baseline Result Validation & Inference Extraction")
    validator = BaselineValidator(device="cpu")
    val_report, samples, metrics_df = validator.validate_all()

    # Save validation artifacts
    metrics_csv = output_dir / "per_modality_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    logger.info(f"Saved verified metrics to {metrics_csv}")

    val_report_path = output_dir / "baseline_validation_report.json"
    with open(val_report_path, "w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2, default=str)
    logger.info(f"Saved validation report to {val_report_path}")

    # 2. Error Analysis & Confidence Distributions
    logger.info("\n>>> Step 2: Error Sample Extraction & Confidence Analysis")
    analyzer = ErrorAnalyzer(sample_predictions=samples, output_dir=output_dir)
    error_df = analyzer.extract_error_samples()
    conf_df, conf_findings = analyzer.analyze_confidence_distributions()
    retina_df, retina_findings = analyzer.analyze_retina_multilabel()
    skin_df, skin_findings = analyzer.analyze_skin_imbalance()
    cm_findings = analyzer.generate_all_confusion_matrices()

    # 3. Histopathology Sanity Check
    logger.info("\n>>> Step 3: Histopathology Sanity & Ceiling Effect Audit")
    sanity_checker = HistopathologySanityChecker(output_dir=output_dir)
    sanity_findings = sanity_checker.run_audit()

    # 4. Generalization Gaps & Cross-Validation Stability
    logger.info("\n>>> Step 4: Generalization Gap & Cross-Validation Analysis")
    gap_analyzer = GapAndCVAnalyzer(validation_report=val_report, output_dir=output_dir)
    gap_findings = gap_analyzer.analyze_generalization_gaps()
    cv_df, cv_findings = gap_analyzer.analyze_cross_validation_stability()

    # 5. Compile Master Report (20 Sections)
    logger.info("\n>>> Step 5: Master Report Compilation")
    builder = Phase6ReportBuilder(
        output_dir=output_dir,
        validation_report=val_report,
        error_df=error_df,
        confidence_findings=conf_findings,
        retina_findings=retina_findings,
        skin_findings=skin_findings,
        sanity_findings=sanity_findings,
        gap_findings=gap_findings,
        cv_findings=cv_findings,
        cm_findings=cm_findings,
    )
    report_path = builder.build_and_save()

    elapsed = time.time() - start_time
    logger.info(f"\n=================================================================")
    logger.info(f"   PHASE 6 PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS   ")
    logger.info(f"   Report: {report_path}                                           ")
    logger.info("=================================================================")


if __name__ == "__main__":
    main()
