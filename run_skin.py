"""
run_skin.py
============
CLI Entrypoint for Module 12: Skin Lesion Cancer Classification Framework.
Runs end-to-end dataset loading, HAM10000 label recovery, dynamic class discovery,
model training, evaluation, export (ONNX/TorchScript), and sample inference with reliability scoring.

Usage:
    python run_skin.py
    python run_skin.py --sample --sample-size 50 --epochs 1
    python run_skin.py --resume
    python run_skin.py --backbone densenet121 --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from skin.datasets.dataset_builder import create_skin_data_loaders
from skin.datasets.dataset_loader import SkinLesionDatasetLoader
from skin.evaluation.evaluator import SkinLesionEvaluator
from skin.export.onnx_export import SkinONNXExporter
from skin.export.torchscript_export import TorchScriptExporter
from skin.inference.predict_skin import SkinLesionPredictor
from skin.models.skin_model import build_skin_model
from skin.reports.clinical_report import ClinicalReportGenerator
from skin.training.trainer import SkinLesionTrainer
from skin.utils.config import SkinLesionConfig
from skin.utils.logger import get_skin_logger

logger = get_skin_logger("run_skin")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Module 12: Skin Lesion Cancer Classification Framework"
    )
    parser.add_argument("--config", type=str, default=None, help="Path to skin_lesion.yaml")
    parser.add_argument("--sample", action="store_true", help="Run in quick sample mode")
    parser.add_argument("--sample-size", type=int, default=50, help="Samples per class in sample mode")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone architecture")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logger.info("=" * 66)
    logger.info("      Module 12: Skin Lesion Cancer Classification Framework       ")
    logger.info("=" * 66)

    # 1. Load Configuration
    config = SkinLesionConfig.load(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.backbone:
        config.model.backbone = args.backbone

    # 2. Dataset Loading, HAM10000 Label Recovery & Dynamic Disease Class Discovery
    loader = SkinLesionDatasetLoader(
        metadata_path=config.data.metadata_path,
        processed_dir=config.data.processed_images_dir,
        ham10000_metadata_path=config.data.ham10000_metadata_path,
        splits_output_path=config.data.splits_output_path,
        label_map_output_path=config.data.label_map_output_path,
        split_ratios=config.data.split_ratios,
        random_seed=42,
    )
    loader.load_and_filter(sample_mode=args.sample, sample_size=args.sample_size)
    splits_df = loader.prepare_splits()
    label_map = loader.label_map
    num_classes = len(label_map)

    logger.info(f"Loaded {num_classes} skin lesion classes: {label_map}")

    # 3. Create PyTorch DataLoaders
    loaders = create_skin_data_loaders(
        splits_df,
        batch_size=config.training.batch_size,
        target_size=tuple(config.data.target_size),
        num_workers=config.data.num_workers if not args.sample else 0,
    )

    if "train" not in loaders or "val" not in loaders:
        logger.error("Missing required train or val DataLoader splits.")
        return 1

    # 4. Build Model Architecture
    model = build_skin_model(config.model, num_classes=num_classes)
    logger.info(f"Built SkinLesionClassifier with backbone: '{config.model.backbone}'")

    # 5. Model Training
    trainer = SkinLesionTrainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config.training,
        save_dir=config.model.saved_model_dir,
        label_map=label_map,
    )
    trainer.fit(resume=args.resume)

    # 6. Evaluation
    test_loader = loaders.get("test", loaders["val"])
    logger.info("Starting test evaluation...")
    evaluator = SkinLesionEvaluator(
        model=model,
        test_loader=test_loader,
        label_map=label_map,
        config=config.evaluation,
    )
    eval_results = evaluator.evaluate()
    evaluator.save_reports(eval_results)

    # 7. Model Export
    logger.info("Exporting production models...")
    SkinONNXExporter.export(
        model=model,
        label_map=label_map,
        export_path=config.export.onnx_path,
        opset_version=config.export.opset_version,
    )
    TorchScriptExporter.export(
        model=model,
        export_path=config.export.torchscript_path,
    )

    # 8. Sample Inference & Clinical Report
    predictor = SkinLesionPredictor(
        model=model,
        label_map=label_map,
        high_threshold=config.reliability.high_threshold,
        moderate_threshold=config.reliability.moderate_threshold,
    )
    test_dataset = test_loader.dataset
    if len(test_dataset) > 0:
        import numpy as np
        from PIL import Image
        sample_img, _ = test_dataset[0]
        sample_pil = Image.fromarray(
            (sample_img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        )
        pred_res = predictor.predict_image(sample_pil)
        logger.info(
            f"Sample inference: '{pred_res.predicted_class}' | "
            f"Reliability: '{pred_res.prediction_reliability}' | "
            f"Requires Manual Review: {pred_res.requires_manual_review} | "
            f"Confidence: {pred_res.confidence*100:.2f}%"
        )
        ClinicalReportGenerator.generate_report(
            pred_res,
            output_path=Path(config.inference.reports_dir) / "skin_report.json",
        )

    acc = eval_results["global_metrics"]["accuracy"]
    f1 = eval_results["global_metrics"]["f1_macro"]
    logger.info(
        f"Module 12 execution complete! "
        f"Test Accuracy: {acc*100:.2f}%, Macro F1: {f1:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
