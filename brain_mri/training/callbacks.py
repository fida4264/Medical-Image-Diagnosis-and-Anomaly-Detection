"""
brain_mri/training/callbacks.py
================================
Re-exports EarlyStopping and ModelCheckpoint from Module 6 training callbacks.
Zero code duplication.
"""

from modality_classifier.training.callbacks import EarlyStopping, ModelCheckpoint

__all__ = ["EarlyStopping", "ModelCheckpoint"]
