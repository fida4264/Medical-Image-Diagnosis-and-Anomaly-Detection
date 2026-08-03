"""
histopathology/training/callbacks.py
=======================================
Re-exports EarlyStopping and ModelCheckpoint from Module 6. Zero duplication.
"""

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["EarlyStopping", "ModelCheckpoint"]
