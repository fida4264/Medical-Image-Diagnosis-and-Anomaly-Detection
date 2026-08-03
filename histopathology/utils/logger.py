"""
histopathology/utils/logger.py
================================
Logger for Module 10 — re-exports setup_logger from Module 6 with histopathology log path.
"""

from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # Module 6 reuse

_LOG_FILE = Path("logs/histopathology/histopathology.log")


def get_histopathology_logger(name: str):
    """Return a logger configured for the histopathology module."""
    return setup_logger(name, log_file=_LOG_FILE)
