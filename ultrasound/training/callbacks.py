"""
ultrasound/training/callbacks.py
================================
Thin re-export of Module 6 callbacks for Module 13.
"""

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["EarlyStopping", "ModelCheckpoint"]
