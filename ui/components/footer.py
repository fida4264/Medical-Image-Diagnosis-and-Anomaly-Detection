"""
footer.py — Institutional Footer Component for Medical AI Platform.
"""

from __future__ import annotations
import gradio as gr
from ui.ui_configuration import (
    PLATFORM_NAME,
    PLATFORM_VERSION,
    DEVELOPER_NAME,
    INSTITUTION,
    DEPARTMENT,
)


def build_footer():
    """Render professional footer banner with credits and institutional links."""
    gr.HTML(f"""
    <div class="med-footer">
      <div class="med-footer-inner">
        <div>
          <div class="med-footer-brand">🩺 {PLATFORM_NAME} {PLATFORM_VERSION}</div>
          <div class="med-footer-sub">
            Developed by <strong>{DEVELOPER_NAME}</strong> · {DEPARTMENT} · {INSTITUTION}
          </div>
        </div>
        <div class="med-footer-links">
          <a href="#tab_home">🏠 Overview</a>
          <a href="#tab_predict">🔬 Diagnostic Pipeline</a>
          <a href="#tab_history">🗄️ Prediction History</a>
          <a href="#tab_benchmark">📊 Global Benchmark</a>
          <a href="#tab_about">ℹ️ Architecture</a>
        </div>
      </div>
      <div class="med-footer-disclaimer">
        ⚠️ <strong>Clinical Disclaimer:</strong> This software is designed for academic research,
        algorithmic verification, and clinical decision support. All AI outputs must be validated by a
        licensed medical professional. Not intended for sole diagnostic use.
        <br/>
        © 2026 {INSTITUTION}. All Rights Reserved. Architecture v0.20.0 (LOCKED).
      </div>
    </div>
    """)
