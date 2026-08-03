"""
header.py — Header component for Medical AI Web App
"""
import gradio as gr
from ui.ui_configuration import PLATFORM_NAME, PLATFORM_SUBTITLE, PLATFORM_VERSION


def build_header():
    """Build sticky medical dashboard header."""
    gr.HTML(
        f"""
        <div class="med-header">
            <div class="med-header-logo">🏥</div>
            <div class="med-header-text">
                <h1>{PLATFORM_NAME}</h1>
                <p>{PLATFORM_SUBTITLE}</p>
            </div>
            <div class="med-header-badge">
                {PLATFORM_VERSION} • System Active
            </div>
        </div>
        """
    )
