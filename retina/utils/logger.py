"""
retina/utils/logger.py
========================
Logger for Module 11 — re-exports setup_logger from Module 6 with retina log path.
Writing to logs/retina/retina.log.
"""

from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # Module 6 reuse

_LOG_FILE = Path("logs/retina/retina.log")


def get_retina_logger(name: str):
    """Return a logger configured for the retina module."""
    return setup_logger(name, log_file=_LOG_FILE)
