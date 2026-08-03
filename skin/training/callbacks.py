"""
skin/training/callbacks.py
===========================
Thin re-export of Module 6 callbacks for Module 12.
"""

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["EarlyStopping", "ModelCheckpoint"]
