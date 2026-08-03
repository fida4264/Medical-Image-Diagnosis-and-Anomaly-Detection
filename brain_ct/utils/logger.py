"""
brain_ct/utils/logger.py
=========================
Logger for Module 9 — re-exports setup_logger from Module 6 with brain_ct log path.
"""

from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # REUSE Module 6

BRAIN_CT_LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "brain_ct.log"


def get_brain_ct_logger(name: str = "brain_ct"):
    """Returns a configured logger writing to logs/brain_ct.log."""
    return setup_logger(name=name, log_file=BRAIN_CT_LOG_FILE)
