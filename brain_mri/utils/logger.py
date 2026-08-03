"""
brain_mri/utils/logger.py
===========================
Logger utility for Module 7 — re-exports setup_logger from Module 6 with brain_mri log path.
Zero duplication.
"""

from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # reuse Module 6

BRAIN_MRI_LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "brain_mri.log"


def get_brain_mri_logger(name: str = "brain_mri"):
    """Returns a configured logger writing to logs/brain_mri.log."""
    return setup_logger(name=name, log_file=BRAIN_MRI_LOG_FILE)
