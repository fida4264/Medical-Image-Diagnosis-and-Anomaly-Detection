"""
ultrasound/utils/logger.py
============================
Logging utility for Module 13: Ultrasound Disease Analysis Framework.
Re-exports Module 6 setup_logger, redirecting output to logs/ultrasound/ultrasound.log.
"""

from __future__ import annotations

import logging
from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # Module 6 reuse


def get_ultrasound_logger(name: str = "ultrasound") -> logging.Logger:
    """Get a logger configured for the ultrasound module."""
    log_file = Path("logs/ultrasound/ultrasound.log")
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return setup_logger(name, log_file=log_file)
