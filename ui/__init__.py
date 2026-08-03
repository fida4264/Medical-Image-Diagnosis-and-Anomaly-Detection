"""
ui package initializer — Module 20: Professional Gradio Web Application
"""

from ui.ui_configuration import *
from ui.theme import MEDICAL_CSS, get_theme
from ui.backend_bridge import *

__all__ = [
    "MEDICAL_CSS",
    "get_theme",
]
