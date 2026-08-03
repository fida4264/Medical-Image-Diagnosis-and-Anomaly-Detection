"""
metadata_manager/__init__.py
=============================
Public API for the metadata_manager package (Module 2).

Exposes the key classes so downstream modules can import cleanly:

    from metadata_manager import MetadataLoader, MetadataStandardizer, ClassMapper
    from metadata_manager import ImageMapper, RegistryGenerator, SummaryGenerator

Author: Medical AI System — Module 2
"""

from metadata_manager.metadata_loader import MetadataLoader
from metadata_manager.metadata_standardizer import MetadataStandardizer
from metadata_manager.class_mapper import ClassMapper
from metadata_manager.image_mapper import ImageMapper
from metadata_manager.registry_generator import RegistryGenerator
from metadata_manager.summary_generator import SummaryGenerator
from metadata_manager.visualization import Visualizer

__all__ = [
    "MetadataLoader",
    "MetadataStandardizer",
    "ClassMapper",
    "ImageMapper",
    "RegistryGenerator",
    "SummaryGenerator",
    "Visualizer",
]

__version__ = "1.0.0"
__module__ = "Module 2 — Dataset Standardization & Metadata Management"
