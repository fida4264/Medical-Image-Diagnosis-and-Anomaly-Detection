"""
pipeline_v2/inference_pipeline.py
=================================
Master V2.0 Hardened Clinical Decision Inference Pipeline.
Coordinates:
1. Input validation & sanitization
2. Modality verification & configuration
3. Preprocessing using established transforms
4. Baseline model forward pass (ResNet-18)
5. Phase 7 confidence calibration & Retina per-label thresholding
6. Phase 8 OOD detection scoring & threshold checking
7. Phase 8 within-distribution reference anomaly scoring
8. Multi-stage quality-gate evaluation & abstention
9. Structured, versioned response generation
10. Privacy-preserving audit logging & provenance
"""

from __future__ import annotations

import io
import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from anomaly_detection.scoring import ReferenceDistributionAnomalyScorer
from ood_detection.feature_distance import KNNDetector, MahalanobisDetector
from ood_detection.modality_adapter import ModalityAdapter
from ood_detection.scoring import OODScorer
from pipeline_v2.audit_logger import AuditLogger
from pipeline_v2.config import (
    AUDIT_LOG_FILE,
    BASELINE_MODELS_DIR,
    CALIBRATION_PARAMS_PATH,
    LEGAL_LIMITATIONS_NOTICE,
    MODALITY_SETTINGS,
    OOD_MODELS_DIR,
    OOD_THRESHOLDS_PATH,
    PIPELINE_VERSION,
    RETINA_THRESHOLDS_PATH,
    ROOT,
    SUPPORTED_MODALITIES,
)
from pipeline_v2.quality_gates import QualityGateEvaluator
from pipeline_v2.schemas import (
    GateOutcome,
    QualityGateResult,
    QualityGateState,
    StructuredInferenceResponse,
)
from pipeline_v2.validator import InputValidationError, InputValidator

logger = logging.getLogger(__name__)


class V2InferencePipeline:
    """
    Unified, hardened inference pipeline integrating Phases 5, 7, 8, and 9.
    """

    def __init__(
        self,
        enforce_ood_rejection: bool = True,
        enforce_calibration: bool = True,
        primary_ood_method: str = "knn",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.enforce_ood_rejection = enforce_ood_rejection
        self.enforce_calibration = enforce_calibration
        self.primary_ood_method = primary_ood_method.lower()

        self.validator = InputValidator()
        self.gate_evaluator = QualityGateEvaluator(
            enforce_ood_rejection=enforce_ood_rejection,
            enforce_calibration=enforce_calibration,
        )
        self.audit_logger = AuditLogger(log_file=AUDIT_LOG_FILE)

        # Lazy cache for models, calibration, and OOD artifacts
        self._adapters: Dict[str, ModalityAdapter] = {}
        self._calibration_params: Optional[Dict[str, Any]] = None
        self._retina_thresholds: Optional[Dict[str, Any]] = None
        self._ood_thresholds: Optional[Dict[str, Any]] = None
        self._mahalanobis_detectors: Dict[str, MahalanobisDetector] = {}
        self._knn_detectors: Dict[str, KNNDetector] = {}
        self._anomaly_scorers: Dict[str, ReferenceDistributionAnomalyScorer] = {}
        self._anomaly_thresholds: Dict[str, float] = {
            "brain_mri": 153.55,
            "brain_ct": 819.15,
            "chest_xray": 55.13,
            "histopathology": 168.15,
            "retina": 133.09,
            "skin": 45.06,
            "ultrasound": 932.94,
        }

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        """Loads Phase 7 calibration and Phase 8 threshold artifacts."""
        if CALIBRATION_PARAMS_PATH.exists():
            try:
                with open(CALIBRATION_PARAMS_PATH, "r", encoding="utf-8") as f:
                    self._calibration_params = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load calibration parameters: {e}")

        if RETINA_THRESHOLDS_PATH.exists():
            try:
                with open(RETINA_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                    self._retina_thresholds = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load Retina thresholds: {e}")

        if OOD_THRESHOLDS_PATH.exists():
            try:
                with open(OOD_THRESHOLDS_PATH, "r", encoding="utf-8") as f:
                    self._ood_thresholds = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load OOD thresholds: {e}")

    def _get_adapter(self, modality: str) -> ModalityAdapter:
        """Retrieves or instantiates cached ModalityAdapter."""
        if modality not in self._adapters:
            adapter = ModalityAdapter(modality=modality, device=str(self.device), use_calibrated=True)
            self._adapters[modality] = adapter
        return self._adapters[modality]

    def _get_ood_detectors(self, modality: str) -> Tuple[Optional[MahalanobisDetector], Optional[KNNDetector]]:
        """Loads fitted Phase 8 Mahalanobis and KNN detectors."""
        if modality not in self._mahalanobis_detectors:
            mah_path = OOD_MODELS_DIR / f"{modality}_mahalanobis.npz"
            if mah_path.exists():
                try:
                    self._mahalanobis_detectors[modality] = MahalanobisDetector.load(mah_path)
                except Exception as e:
                    logger.warning(f"Failed to load Mahalanobis for {modality}: {e}")

        if modality not in self._knn_detectors:
            knn_path = OOD_MODELS_DIR / f"{modality}_knn.npz"
            if knn_path.exists():
                try:
                    self._knn_detectors[modality] = KNNDetector.load(knn_path)
                except Exception as e:
                    logger.warning(f"Failed to load KNN for {modality}: {e}")

        return self._mahalanobis_detectors.get(modality), self._knn_detectors.get(modality)

    def _get_anomaly_scorer(self, modality: str) -> Optional[ReferenceDistributionAnomalyScorer]:
        """Loads fitted ReferenceDistributionAnomalyScorer."""
        if modality not in self._anomaly_scorers:
            anom_path = OOD_MODELS_DIR / f"{modality}_normal_anomaly.npz"
            if anom_path.exists():
                try:
                    self._anomaly_scorers[modality] = ReferenceDistributionAnomalyScorer.load(anom_path)
                except Exception as e:
                    logger.warning(f"Failed to load anomaly scorer for {modality}: {e}")
        return self._anomaly_scorers.get(modality)

    def predict(
        self,
        image_input: Union[bytes, str, Path, Image.Image, io.BytesIO],
        modality: Optional[str] = None,
        request_id: Optional[str] = None,
        patient_id: Optional[str] = None,
        ood_method_override: Optional[str] = None,
    ) -> StructuredInferenceResponse:
        """
        Executes the hardened 10-step inference pipeline.
        """
        req_id = request_id or str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc).isoformat()
        ood_method = (ood_method_override or self.primary_ood_method).lower()

        # Raw payload bytes for hashing provenance
        raw_bytes: Optional[bytes] = None
        if isinstance(image_input, (bytes, io.BytesIO)):
            raw_bytes = image_input.getvalue() if isinstance(image_input, io.BytesIO) else image_input
        elif isinstance(image_input, (str, Path)):
            p = Path(image_input)
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        raw_bytes = f.read()
                except Exception:
                    pass

        # -------------------------------------------------------------
        # Stage A & B: Input and Modality Validation
        # -------------------------------------------------------------
        sanitized_img: Optional[Image.Image] = None
        clean_modality: str = "unknown"
        input_meta: Dict[str, Any] = {}
        input_valid = True
        input_error: Optional[str] = None
        modality_valid = True
        modality_error: Optional[str] = None

        try:
            sanitized_img, input_meta = self.validator.validate_image_input(image_input)
        except InputValidationError as e:
            input_valid = False
            input_error = str(e)
        except Exception as e:
            input_valid = False
            input_error = f"Unexpected input decoding error: {e}"

        try:
            clean_modality = self.validator.validate_modality(modality)
        except InputValidationError as e:
            modality_valid = False
            modality_error = str(e)
        except Exception as e:
            modality_valid = False
            modality_error = f"Modality verification error: {e}"

        # If input or modality validation failed, early exit with rejection
        if not input_valid or not modality_valid:
            gate_res = self.gate_evaluator.evaluate_gates(
                input_valid=input_valid,
                input_error=input_error,
                modality_valid=modality_valid,
                modality_error=modality_error,
                model_available=False,
            )
            self.audit_logger.record_inference(
                request_id=req_id,
                modality=clean_modality,
                model_identifier="none",
                checkpoint_identifier="none",
                quality_gate_result=gate_res,
                image_bytes=raw_bytes,
                patient_pseudonym=patient_id,
                error_code="VALIDATION_FAILED",
            )
            return StructuredInferenceResponse(
                request_id=req_id,
                pipeline_version=PIPELINE_VERSION,
                inference_timestamp=timestamp,
                modality=clean_modality,
                model_identifier="none",
                checkpoint_identifier="none",
                prediction_status=gate_res.overall_status,
                is_abstained=True,
                ood_method=ood_method,
                ood_status="UNAVAILABLE",
                quality_gate_results=gate_res,
                rejection_or_review_reasons=gate_res.rejection_reasons,
                preprocessing_configuration={},
                calibration_configuration={},
                limitations_notice=LEGAL_LIMITATIONS_NOTICE,
            )

        # -------------------------------------------------------------
        # Stage C & D: Preprocessing & Baseline Model Forward Pass
        # -------------------------------------------------------------
        cfg = MODALITY_SETTINGS[clean_modality]
        model_available = True
        model_error: Optional[str] = None
        adapter: Optional[ModalityAdapter] = None

        try:
            adapter = self._get_adapter(clean_modality)
            logits_1d, feats_1d = adapter.process_image(sanitized_img)
        except Exception as e:
            model_available = False
            model_error = f"Model execution error for {clean_modality}: {e}"
            logits_1d = None
            feats_1d = None

        if not model_available or logits_1d is None:
            gate_res = self.gate_evaluator.evaluate_gates(
                input_valid=True,
                modality_valid=True,
                model_available=False,
                model_error=model_error,
            )
            self.audit_logger.record_inference(
                request_id=req_id,
                modality=clean_modality,
                model_identifier="resnet18",
                checkpoint_identifier=str(cfg["checkpoint"]),
                quality_gate_result=gate_res,
                image_bytes=raw_bytes,
                patient_pseudonym=patient_id,
                error_code="INFERENCE_FAILED",
            )
            return StructuredInferenceResponse(
                request_id=req_id,
                pipeline_version=PIPELINE_VERSION,
                inference_timestamp=timestamp,
                modality=clean_modality,
                model_identifier="resnet18",
                checkpoint_identifier=str(cfg["checkpoint"]),
                prediction_status=QualityGateState.INFERENCE_ERROR,
                is_abstained=True,
                ood_method=ood_method,
                ood_status="UNAVAILABLE",
                quality_gate_results=gate_res,
                rejection_or_review_reasons=gate_res.rejection_reasons,
                preprocessing_configuration={"input_resolution": [224, 224], "channels": 3},
                calibration_configuration={},
                limitations_notice=LEGAL_LIMITATIONS_NOTICE,
            )

        # -------------------------------------------------------------
        # Stage E: Phase 7 Confidence Calibration & Thresholding
        # -------------------------------------------------------------
        task = cfg["task"]
        class_names = cfg["classes"]
        raw_logits_dict = {c: round(float(logits_1d[i]), 4) for i, c in enumerate(class_names)}
        calibrated_probs_dict: Optional[Dict[str, float]] = None
        retina_predictions_dict: Optional[Dict[str, int]] = None
        retina_thresholds_used_dict: Optional[Dict[str, float]] = None
        temperature_used = 1.0
        calibrated = False

        if task == "multilabel":
            # Multi-label Retina: independent sigmoid heads
            sigmoids = 1.0 / (1.0 + np.exp(-logits_1d))
            calibrated_probs_dict = {c: round(float(sigmoids[i]), 4) for i, c in enumerate(class_names)}

            # Apply per-label Phase 7 optimal thresholds
            retina_thresholds_used_dict = {}
            retina_predictions_dict = {}
            phase7_thresh = (self._retina_thresholds or {}).get("retina_thresholds", {})

            for i, c in enumerate(class_names):
                tau = float(phase7_thresh.get(c, 0.5))
                retina_thresholds_used_dict[c] = tau
                retina_predictions_dict[c] = int(sigmoids[i] > tau)

            predicted_class = "multiple_or_normal"
            top_prob = float(np.max(sigmoids))
            calibrated = True

        else:
            # Multiclass modalities: apply Phase 7 temperature scaling
            temp_cfg = (self._calibration_params or {}).get(clean_modality, {})
            temperature_used = float(temp_cfg.get("temperature", 1.0))
            scaled_logits = logits_1d / temperature_used

            # Softmax
            exp_z = np.exp(scaled_logits - np.max(scaled_logits))
            probs = exp_z / np.sum(exp_z)
            calibrated_probs_dict = {c: round(float(probs[i]), 4) for i, c in enumerate(class_names)}

            top_idx = int(np.argmax(probs))
            predicted_class = class_names[top_idx]
            top_prob = float(probs[top_idx])
            calibrated = True

        # -------------------------------------------------------------
        # Stage F: Phase 8 Out-of-Distribution (OOD) Scoring
        # -------------------------------------------------------------
        mah_det, knn_det = self._get_ood_detectors(clean_modality)
        scorer = OODScorer(mahalanobis_detector=mah_det, knn_detector=knn_det)

        ood_score: Optional[float] = None
        ood_threshold: Optional[float] = None
        ood_status = "UNAVAILABLE"

        try:
            scores_batch = scorer.compute_score(
                method=ood_method,
                logits=logits_1d[np.newaxis, :],
                features=feats_1d[np.newaxis, :],
                task=task,
                temperature=temperature_used,
            )
            ood_score = float(scores_batch[0])
            mod_thresh = (self._ood_thresholds or {}).get(clean_modality, {}).get(ood_method, {})
            ood_threshold = float(mod_thresh.get("chosen_operating_threshold", 0.5))

            # HIGHER = MORE OOD
            if ood_score > ood_threshold:
                ood_status = "POTENTIALLY_OOD"
            else:
                ood_status = "WITHIN_DISTRIBUTION"
        except Exception as e:
            logger.warning(f"OOD scoring failed for {clean_modality}: {e}")

        # -------------------------------------------------------------
        # Stage G: Phase 8 Reference Anomaly Scoring
        # -------------------------------------------------------------
        anom_scorer = self._get_anomaly_scorer(clean_modality)
        anomaly_score: Optional[float] = None
        anomaly_threshold: Optional[float] = self._anomaly_thresholds.get(clean_modality)
        anomaly_status = "UNAVAILABLE"

        if anom_scorer is not None:
            try:
                anom_batch = anom_scorer.score(feats_1d[np.newaxis, :])
                anomaly_score = float(anom_batch[0])
                if anomaly_threshold is not None:
                    if anomaly_score > anomaly_threshold:
                        anomaly_status = "POTENTIAL_ABNORMALITY"
                    else:
                        anomaly_status = "NORMAL_REFERENCE_ALIGNED"
            except Exception as e:
                logger.warning(f"Anomaly scoring failed for {clean_modality}: {e}")

        # -------------------------------------------------------------
        # Stage H: Quality-Gate Evaluation
        # -------------------------------------------------------------
        gate_res = self.gate_evaluator.evaluate_gates(
            input_valid=True,
            modality_valid=True,
            model_available=True,
            logits=logits_1d,
            probabilities=np.array(list(calibrated_probs_dict.values())),
            calibrated=calibrated,
            ood_score=ood_score,
            ood_threshold=ood_threshold,
            anomaly_score=anomaly_score,
            anomaly_threshold=anomaly_threshold,
        )

        is_abstained = gate_res.overall_status in (
            QualityGateState.REJECTED_OOD,
            QualityGateState.REJECTED_INVALID_INPUT,
            QualityGateState.REJECTED_UNSUPPORTED_MODALITY,
            QualityGateState.INFERENCE_ERROR,
        )

        # -------------------------------------------------------------
        # Stage I: Audit Logging & Provenance
        # -------------------------------------------------------------
        self.audit_logger.record_inference(
            request_id=req_id,
            modality=clean_modality,
            model_identifier="resnet18",
            checkpoint_identifier=str(cfg["checkpoint"]),
            quality_gate_result=gate_res,
            image_bytes=raw_bytes,
            patient_pseudonym=patient_id,
            ood_method=ood_method,
            ood_score=ood_score,
            ood_threshold=ood_threshold,
            anomaly_score=anomaly_score,
            anomaly_threshold=anomaly_threshold,
            calibration_artifact_id=f"temp_{temperature_used:.4f}" if task != "multilabel" else "retina_f1_thresholds",
        )

        # -------------------------------------------------------------
        # Stage J: Structured Response Assembly
        # -------------------------------------------------------------
        final_predicted_class = None if is_abstained else predicted_class
        final_confidence = None if is_abstained else round(top_prob, 4)

        return StructuredInferenceResponse(
            request_id=req_id,
            pipeline_version=PIPELINE_VERSION,
            inference_timestamp=timestamp,
            modality=clean_modality,
            model_identifier="resnet18",
            checkpoint_identifier=str(cfg["checkpoint"]),
            prediction_status=gate_res.overall_status,
            is_abstained=is_abstained,
            predicted_class=final_predicted_class,
            confidence_score=final_confidence,
            raw_model_logits=raw_logits_dict,
            calibrated_probabilities=calibrated_probs_dict,
            retina_label_predictions=retina_predictions_dict,
            retina_thresholds_used=retina_thresholds_used_dict,
            ood_method=ood_method,
            ood_score=round(ood_score, 4) if ood_score is not None else None,
            ood_threshold=round(ood_threshold, 4) if ood_threshold is not None else None,
            ood_status=ood_status,
            anomaly_method="mahalanobis_normal_reference" if anomaly_score is not None else None,
            anomaly_score=round(anomaly_score, 4) if anomaly_score is not None else None,
            anomaly_threshold=round(anomaly_threshold, 4) if anomaly_threshold is not None else None,
            anomaly_status=anomaly_status,
            quality_gate_results=gate_res,
            rejection_or_review_reasons=gate_res.rejection_reasons or gate_res.review_reasons,
            preprocessing_configuration={
                "input_resolution": [224, 224],
                "channels": 3,
                "interpolation": "bilinear",
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            },
            calibration_configuration={
                "task": task,
                "method": "temperature_scaling" if task != "multilabel" else "per_label_f1_thresholding",
                "temperature": temperature_used if task != "multilabel" else None,
            },
            limitations_notice=LEGAL_LIMITATIONS_NOTICE,
        )


# Singleton instance accessor
_pipeline_instance: Optional[V2InferencePipeline] = None


def get_pipeline_v2() -> V2InferencePipeline:
    """Returns the singleton V2InferencePipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = V2InferencePipeline()
    return _pipeline_instance
