"""
chest_xray/utils/logger.py
============================
Logger for Module 8 — re-exports setup_logger from Module 6 with chest_xray log path.
"""

from pathlib import Path

from modality_classifier.utils.logging import setup_logger  # reuse Module 6

CHEST_XRAY_LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "chest_xray.log"


def get_chest_xray_logger(name: str = "chest_xray"):
    """Returns a configured logger writing to logs/chest_xray.log."""
    return setup_logger(name=name, log_file=CHEST_XRAY_LOG_FILE)
