"""
modality_classifier/utils/logging.py
=====================================
Logger setup utility writing to console and file.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

LOGS_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_FILE = LOGS_DIR / "modality_classifier.log"


def setup_logger(name: str = "modality_classifier", log_file: Optional[Path] = None, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if logger.handlers:
        return logger

    target_log_file = log_file or LOG_FILE
    target_log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Console
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(target_log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger
