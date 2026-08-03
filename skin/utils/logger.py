"""
skin/utils/logger.py
======================
Logging utility for Module 12: Skin Lesion Cancer Classification Framework.
Re-exports Module 6 setup_logger, redirecting output to logs/skin/skin.log.
"""

from __future__ import annotations

import logging
from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # Module 6 reuse


def get_skin_logger(name: str = "skin_lesion") -> logging.Logger:
    """Get a logger configured for the skin lesion module."""
    log_file = Path("logs/skin/skin.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return setup_logger(name, log_file=log_file)
