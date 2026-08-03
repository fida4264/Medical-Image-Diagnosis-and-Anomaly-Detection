"""
demo_mode.py — Automated Demo Mode for the Medical AI Platform.

Lets reviewers/evaluators trigger a complete automated pipeline run
using synthetic demo images for any supported modality, without uploading
a real scan.
"""

from __future__ import annotations
import os
import tempfile
import logging
from pathlib import Path
import gradio as gr
import numpy as np

logger = logging.getLogger("MedicalAI.Demo")

# Modality display options for demo mode
DEMO_MODALITIES = [
    ("Brain MRI",       "brain_mri",       "🧠"),
    ("Chest X-Ray",     "chest_xray",      "🫁"),
    ("Brain CT",        "brain_ct",        "🧬"),
    ("Histopathology",  "histopathology",  "🔬"),
    ("Retina",          "retina",          "👁️"),
    ("Skin Lesion",     "skin",            "🩺"),
    ("Ultrasound",      "ultrasound",      "📡"),
]


def _generate_demo_image(modality_key: str) -> str:
    """
    Generate a synthetic placeholder image for demo purposes.
    Returns a file path to a temporary PNG.
    """
    try:
        from PIL import Image, ImageDraw, ImageFilter
    except ImportError:
        return _save_numpy_image(modality_key)

    # Color schemes per modality
    schemes = {
        "brain_mri":       ((10, 15, 30),   (100, 160, 220)),
        "chest_xray":      ((20, 20, 20),   (180, 190, 200)),
        "brain_ct":        ((5, 8, 12),     (80, 130, 170)),
        "histopathology":  ((30, 10, 30),   (160, 80, 140)),
        "retina":          ((5, 20, 5),     (20, 140, 60)),
        "skin":            ((60, 35, 20),   (180, 130, 90)),
        "ultrasound":      ((5, 5, 15),     (50, 80, 130)),
    }
    bg, fg = schemes.get(modality_key, ((10, 15, 30), (100, 160, 220)))

    img = Image.new("RGB", (256, 256), bg)
    draw = ImageDraw.Draw(img)

    # Draw some circles/ellipses to mimic medical imagery
    import random
    rng = random.Random(hash(modality_key) % 1000)
    for _ in range(20):
        x, y = rng.randint(20, 236), rng.randint(20, 236)
        r = rng.randint(5, 40)
        alpha = rng.uniform(0.3, 0.9)
        col = tuple(int(c * alpha + b * (1 - alpha)) for c, b in zip(fg, bg))
        draw.ellipse([x - r, y - r, x + r, y + r], fill=col, outline=None)

    # Gaussian blur to mimic scan
    img = img.filter(ImageFilter.GaussianBlur(radius=3))

    with tempfile.NamedTemporaryFile(suffix=f"_demo_{modality_key}.png", delete=False) as f:
        img.save(f.name)
        return f.name


def _save_numpy_image(modality_key: str) -> str:
    """Fallback: save a numpy-generated image."""
    arr = np.random.randint(30, 180, (256, 256, 3), dtype=np.uint8)
    with tempfile.NamedTemporaryFile(suffix=f"_demo_{modality_key}.png", delete=False) as f:
        try:
            from PIL import Image
            Image.fromarray(arr).save(f.name)
        except Exception:
            f.write(arr.tobytes())
        return f.name


def build_demo_mode_section():
    """
    Build the Demo Mode UI section.
    Returns the Gradio components needed to wire demo into the prediction pipeline.
    """
    with gr.Column():
        gr.HTML("""
        <div class="demo-banner">
          <span class="demo-banner-icon">🎬</span>
          <div class="demo-banner-text">
            <strong>Demo Mode — Automated Pipeline Showcase</strong>
            <span>Select a medical imaging modality to run a complete automated pipeline
            demonstration with a synthetic scan. No real image required.</span>
          </div>
        </div>
        """)

        gr.HTML("""
        <div class="demo-selector">
          <div style="font-size:12px;font-weight:700;color:#94a3b8;
               text-transform:uppercase;letter-spacing:1px;margin-bottom:12px;">
            Select Imaging Modality
          </div>
        """)

        modality_radio = gr.Radio(
            choices=[f"{icon} {label}" for label, _, icon in DEMO_MODALITIES],
            label="",
            value=f"{DEMO_MODALITIES[0][2]} {DEMO_MODALITIES[0][0]}",
            elem_classes=["demo-radio"],
        )

        gr.HTML("</div>")  # close demo-selector

        demo_run_btn = gr.Button(
            "▶️ Run Automated Demo",
            variant="primary",
            elem_classes=["btn-primary"],
        )
        demo_status = gr.HTML(
            value="<div style='color:#94a3b8;font-size:12px;'>Click above to start the demo.</div>"
        )

    return modality_radio, demo_run_btn, demo_status


def run_demo_pipeline(modality_radio_value: str):
    """
    Generate a synthetic demo image and run the full prediction pipeline.
    Returns the full prediction result tuple identical to handle_analysis.
    """
    from ui.backend_bridge import run_prediction, generate_report

    # Extract modality key from radio value
    mod_key = None
    for label, key, icon in DEMO_MODALITIES:
        if label in modality_radio_value:
            mod_key = key
            break
    if mod_key is None:
        mod_key = "brain_mri"

    img_path = _generate_demo_image(mod_key)
    patient_id = "DEMO-PT-2026"

    success, res, err = run_prediction(img_path, mod_key, patient_id)
    return success, res, err, img_path
