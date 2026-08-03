"""
run_brain_ct.py
===============
CLI Entrypoint for Module 9: Brain CT Disease Analysis Framework.
Runs end-to-end dataset loading, disease class discovery, model training, evaluation,
model export (ONNX/TorchScript), and sample inference.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from brain_ct.datasets.dataset_builder import create_brain_ct_data_loaders
from brain_ct.datasets.dataset_loader import BrainCTDatasetLoader
from brain_ct.evaluation.evaluator import BrainCTEvaluator
from brain_ct.export.onnx_export import BrainCTONNXExporter
from brain_ct.export.torchscript_export import TorchScriptExporter
from brain_ct.inference.predict_brain_ct import BrainCTPredictor
from brain_ct.models.brain_ct_model import build_brain_ct_model
from brain_ct.reports.clinical_report import ClinicalReportGenerator
from brain_ct.training.trainer import BrainCTTrainer
from brain_ct.utils.config import BrainCTConfig
from brain_ct.utils.logger import get_brain_ct_logger

logger = get_brain_ct_logger("run_brain_ct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Module 9: Brain CT Disease Analysis Framework")
    parser.add_argument("--config", type=str, default=None, help="Path to brain_ct.yaml")
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
    logger.info("        Module 9: Brain CT Disease Analysis Framework         ")
    logger.info("=" * 66)

    # 1. Load Configuration
    config = BrainCTConfig.load(args.config)
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.backbone:
        config.model.backbone = args.backbone

    # 2. Dataset Loading & Disease Class Discovery
    loader = BrainCTDatasetLoader(
        metadata_path=config.data.metadata_path,
        processed_dir=config.data.processed_images_dir,
        splits_output_path=config.data.splits_output_path,
        label_map_output_path=config.data.label_map_output_path,
        split_ratios=config.data.split_ratios,
        random_seed=config.data.random_seed,
    )
    df = loader.load_and_filter(sample_mode=args.sample, sample_size=args.sample_size)
    splits_df = loader.prepare_splits()
    label_map = loader.label_map
    num_classes = len(label_map)

    logger.info(f"Loaded {num_classes} Brain CT disease classes: {label_map}")

    # 3. Create PyTorch DataLoaders
    loaders = create_brain_ct_data_loaders(
        splits_df,
        batch_size=config.training.batch_size,
        target_size=config.data.target_size,
        num_workers=config.data.num_workers if not args.sample else 0,
    )

    if "train" not in loaders or "val" not in loaders:
        logger.error("Missing required train or val DataLoader splits.")
        return 1

    # 4. Build Model Architecture
    model = build_brain_ct_model(config.model, num_classes=num_classes)
    logger.info(f"Built BrainCTClassifier model with backbone: '{config.model.backbone}'")

    # 5. Model Training
    trainer = BrainCTTrainer(
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
    evaluator = BrainCTEvaluator(
        model=model,
        test_loader=test_loader,
        label_map=label_map,
        config=config.evaluation,
    )
    eval_results = evaluator.evaluate()
    json_report, md_report = evaluator.save_reports(eval_results)

    # 7. Model Export
    logger.info("Exporting production models...")
    onnx_path = BrainCTONNXExporter.export(
        model=model,
        label_map=label_map,
        export_path=config.export.onnx_path,
        opset_version=config.export.opset_version,
    )
    torchscript_path = TorchScriptExporter.export(
        model=model,
        export_path=config.export.torchscript_path,
    )

    # 8. Sample Inference & Clinical Report Generation
    predictor = BrainCTPredictor(model=model, label_map=label_map)
    test_dataset = test_loader.dataset
    if len(test_dataset) > 0:
        sample_img, _ = test_dataset[0]
        import numpy as np
        from PIL import Image
        sample_pil = Image.fromarray((sample_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        pred_res = predictor.predict_image(sample_pil)
        ClinicalReportGenerator.generate_report(
            pred_res,
            output_path=Path(config.inference.reports_dir) / "brain_ct_report.json",
        )

    acc = eval_results["global_metrics"]["accuracy"]
    f1 = eval_results["global_metrics"]["f1_macro"]
    logger.info(f"Module 9 execution complete! Test Accuracy: {acc*100:.2f}%, Macro F1: {f1:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
