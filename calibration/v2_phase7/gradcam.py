"""
calibration/v2_phase7/gradcam.py
=================================
Gradient-weighted Class Activation Mapping (Grad-CAM) for ResNet-18.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
via Gradient-based Localization," ICCV 2017.

IMPORTANT LIMITATIONS (required by Phase 7 spec):
  - Grad-CAM is an explainability visualization aid ONLY.
  - Heatmaps do NOT prove that the model understands pathology.
  - Heatmaps do NOT constitute clinical reasoning or clinical validation.
  - Terminology used: "attention map", "saliency visualization", "activation region"

Architecture:
  For ResNet-18 in V2BaselineModel:
    model.backbone = nn.Sequential(conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4)
    Grad-CAM target layer = model.backbone[-1]  (layer4, output: B×512×7×7)

Generates overlays for 4 categories per modality:
  A. Correct   high-confidence predictions
  B. Correct   low-confidence predictions
  C. Incorrect high-confidence predictions (highest clinical relevance)
  D. Incorrect low-confidence predictions
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

HIGH_CONF_THRESHOLD = 0.80
LOW_CONF_THRESHOLD = 0.50
SAMPLES_PER_CATEGORY = 5  # Per modality, per category


class GradCAMHooks:
    """
    Attaches forward and backward hooks to the target convolutional layer.
    Records feature maps and gradients during a single forward+backward pass.
    """

    def __init__(self, layer: torch.nn.Module):
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = layer.register_forward_hook(self._save_activations)
        self._backward_hook = layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove(self):
        """Removes hooks from the model."""
        self._forward_hook.remove()
        self._backward_hook.remove()


class GradCAMEngine:
    """
    Grad-CAM generator for V2BaselineModel (ResNet-18 backbone).
    Does NOT modify model weights.
    """

    def __init__(self, model: torch.nn.Module, device: str = "cpu"):
        """
        Args:
            model: V2BaselineModel instance (eval mode)
            device: Torch device string
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()

        # Determine Grad-CAM target layer
        # For ResNet-18: model.backbone[-1] = layer4
        self.target_layer = model.backbone[-1]

    @torch.enable_grad()
    def generate(
        self,
        image_tensor: torch.Tensor,
        class_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap for a single image.

        Args:
            image_tensor: Preprocessed image tensor, shape (1, 3, H, W)
            class_idx: Target class index. If None, uses argmax of predicted logits.

        Returns:
            Normalized heatmap array, shape (H, W), values in [0, 1]
        """
        image_tensor = image_tensor.to(self.device)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Attach hooks
        hooks = GradCAMHooks(self.target_layer)

        try:
            # Forward pass
            self.model.zero_grad()
            logits = self.model(image_tensor)

            # Determine target class
            if class_idx is None:
                class_idx = int(logits.argmax(dim=1).item())

            # Backward pass for target class
            target_score = logits[0, class_idx]
            target_score.backward()

            # Compute Grad-CAM
            activations = hooks.activations  # (1, C, H', W')
            gradients = hooks.gradients      # (1, C, H', W')

            if activations is None or gradients is None:
                logger.warning("Hooks did not capture activations or gradients.")
                return np.zeros((224, 224), dtype=np.float32)

            # Global average pooling of gradients → channel weights
            alpha = gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

            # Weighted combination of feature maps
            cam = (alpha * activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
            cam = F.relu(cam)

            # Upsample to input spatial size
            h, w = image_tensor.shape[2], image_tensor.shape[3]
            cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
            cam = cam.squeeze().cpu().numpy()

            # Normalize to [0, 1]
            cam_min, cam_max = cam.min(), cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = np.zeros_like(cam)

            return cam.astype(np.float32)

        finally:
            hooks.remove()
            self.model.zero_grad()

    def generate_multilabel(
        self,
        image_tensor: torch.Tensor,
        label_idx: int,
    ) -> np.ndarray:
        """
        Grad-CAM for a specific sigmoid label in multi-label classification (Retina).

        Args:
            image_tensor: Preprocessed image tensor, shape (1, 3, H, W)
            label_idx: Index of the target sigmoid output

        Returns:
            Normalized heatmap array, shape (H, W), values in [0, 1]
        """
        return self.generate(image_tensor, class_idx=label_idx)

    def overlay_heatmap(
        self,
        original_image: Image.Image,
        heatmap: np.ndarray,
        alpha: float = 0.45,
        colormap: str = "jet",
    ) -> Image.Image:
        """
        Blends a Grad-CAM heatmap over the original image using a colormap.

        Args:
            original_image: Original PIL image (RGB)
            heatmap: Normalized heatmap array, shape (H, W), values in [0, 1]
            alpha: Opacity of the heatmap overlay (0=original, 1=heatmap only)
            colormap: Colormap name ('jet' standard for Grad-CAM)

        Returns:
            Blended PIL image (RGB)
        """
        import matplotlib
        try:
            cmap = matplotlib.colormaps[colormap]
        except (AttributeError, KeyError):
            import matplotlib.cm as cm
            cmap = cm.get_cmap(colormap)

        # Resize heatmap to match image
        w, h = original_image.size
        heatmap_resized = np.array(
            Image.fromarray((heatmap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
        ) / 255.0

        colored = (cmap(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
        colored_pil = Image.fromarray(colored, mode="RGB")

        # Blend
        original_rgb = original_image.convert("RGB")
        blended = Image.blend(original_rgb, colored_pil, alpha=alpha)
        return blended

    def process_sample_batch(
        self,
        sample_records: List[Dict[str, Any]],
        transform,
        output_dir: Path,
        modality: str,
        task: str = "multiclass",
        samples_per_category: int = SAMPLES_PER_CATEGORY,
    ) -> List[Dict[str, Any]]:
        """
        Generates Grad-CAM overlays for a batch of error/correct samples.

        Organizes samples into 4 categories:
          A: correct   + high confidence  (score >= HIGH_CONF_THRESHOLD)
          B: correct   + low confidence   (score < LOW_CONF_THRESHOLD)
          C: incorrect + high confidence  (score >= HIGH_CONF_THRESHOLD)
          D: incorrect + low confidence   (score < LOW_CONF_THRESHOLD)

        Args:
            sample_records: List of dicts with image_path, true_label,
                            predicted_label, prediction_score, correct
            transform: Torchvision transform to apply to images
            output_dir: Directory to save overlay images
            modality: Modality name for subdirectory
            task: Task type ('multiclass', 'binary', 'multilabel')
            samples_per_category: Max samples per category

        Returns:
            List of analysis record dicts (for gradcam_analysis.csv)
        """
        mod_dir = Path(output_dir) / modality
        mod_dir.mkdir(parents=True, exist_ok=True)

        # Categorize samples
        categories = {
            "correct_high_conf": [],
            "correct_low_conf": [],
            "incorrect_high_conf": [],
            "incorrect_low_conf": [],
        }

        for rec in sample_records:
            is_correct = bool(rec.get("correct", False))
            score = float(rec.get("prediction_score", 0.5))
            if is_correct and score >= HIGH_CONF_THRESHOLD:
                categories["correct_high_conf"].append(rec)
            elif is_correct and score < LOW_CONF_THRESHOLD:
                categories["correct_low_conf"].append(rec)
            elif not is_correct and score >= HIGH_CONF_THRESHOLD:
                categories["incorrect_high_conf"].append(rec)
            elif not is_correct and score < LOW_CONF_THRESHOLD:
                categories["incorrect_low_conf"].append(rec)

        analysis_records = []

        for cat_name, cat_samples in categories.items():
            selected = cat_samples[:samples_per_category]
            for i, rec in enumerate(selected):
                img_path = rec.get("image_path", "")
                true_label = rec.get("true_label", "unknown")
                pred_label = rec.get("predicted_label", "unknown")
                score = float(rec.get("prediction_score", 0.0))
                is_correct = bool(rec.get("correct", False))
                image_id = rec.get("image_id", f"{cat_name}_{i}")

                try:
                    # Load and transform image
                    with Image.open(img_path) as img:
                        img_rgb = img.convert("RGB")
                        img_original = img_rgb.copy()
                    img_tensor = transform(img_rgb).unsqueeze(0)

                    # Determine class index for Grad-CAM
                    if task == "multilabel":
                        # For Retina: use predicted label with highest probability
                        with torch.no_grad():
                            probs = torch.sigmoid(self.model(img_tensor.to(self.device)))
                            class_idx = int(probs.argmax(dim=1).item())
                        heatmap = self.generate_multilabel(img_tensor, class_idx)
                    else:
                        with torch.no_grad():
                            logits = self.model(img_tensor.to(self.device))
                            pred_idx = int(logits.argmax(dim=1).item())
                        heatmap = self.generate(img_tensor, class_idx=pred_idx)

                    # Save overlay
                    overlay = self.overlay_heatmap(img_original, heatmap)
                    safe_id = str(image_id).replace("/", "_").replace("\\", "_")
                    out_name = f"{cat_name}_{i:02d}_{safe_id[:40]}.png"
                    overlay_path = mod_dir / out_name
                    overlay.save(overlay_path)

                    analysis_records.append({
                        "modality": modality,
                        "image_id": str(image_id),
                        "image_path": str(img_path),
                        "true_label": str(true_label),
                        "predicted_label": str(pred_label),
                        "prediction_score": round(score, 4),
                        "correct": is_correct,
                        "category": cat_name,
                        "overlay_path": str(overlay_path),
                        "gradcam_generated": True,
                        "error": None,
                    })
                    logger.debug(f"[gradcam/{modality}/{cat_name}] Saved: {out_name}")

                except Exception as e:
                    logger.warning(f"[gradcam/{modality}] Failed for {img_path}: {e}")
                    analysis_records.append({
                        "modality": modality,
                        "image_id": str(rec.get("image_id", "?")),
                        "image_path": str(img_path),
                        "true_label": str(true_label),
                        "predicted_label": str(pred_label),
                        "prediction_score": float(rec.get("prediction_score", 0)),
                        "correct": bool(rec.get("correct", False)),
                        "category": cat_name,
                        "overlay_path": None,
                        "gradcam_generated": False,
                        "error": str(e),
                    })

        logger.info(
            f"[gradcam/{modality}] Generated {sum(1 for r in analysis_records if r['gradcam_generated'])} "
            f"overlays across {len(categories)} categories"
        )
        return analysis_records
