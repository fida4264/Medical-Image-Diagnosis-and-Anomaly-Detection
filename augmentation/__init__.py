"""
augmentation/__init__.py
==========================
Public API for the augmentation package (Module 4).

Usage
-----
    from augmentation import AugmentationFactory, BaseAugmenter
    from augmentation import OfflineGenerator, OnlineGenerator

Author: Medical AI System — Module 4
"""

from augmentation.base.base_augmenter import BaseAugmenter
from augmentation.factory.augmentation_factory import AugmentationFactory
from augmentation.offline.offline_generator import OfflineGenerator
from augmentation.online.online_generator import OnlineGenerator
from augmentation.policies.brain_mri_policy import BrainMRIAugmenter
from augmentation.policies.brain_ct_policy import BrainCTAugmenter
from augmentation.policies.chest_xray_policy import ChestXRayAugmenter
from augmentation.policies.retina_policy import RetinaAugmenter
from augmentation.policies.skin_policy import SkinAugmenter
from augmentation.policies.ultrasound_policy import UltrasoundAugmenter
from augmentation.policies.histopathology_policy import HistopathologyAugmenter

__all__ = [
    "BaseAugmenter",
    "AugmentationFactory",
    "OfflineGenerator",
    "OnlineGenerator",
    "BrainMRIAugmenter",
    "BrainCTAugmenter",
    "ChestXRayAugmenter",
    "RetinaAugmenter",
    "SkinAugmenter",
    "UltrasoundAugmenter",
    "HistopathologyAugmenter",
]

__version__ = "1.0.0"
__module__ = "Module 4 — Medical Image Augmentation Framework"
