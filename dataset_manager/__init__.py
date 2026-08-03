"""
__init__.py
===========
Public API of the dataset_manager package.

Importing from dataset_manager directly exposes the most commonly used classes
so that future modules can write clean, short import statements.

Usage
-----
    from dataset_manager import DatasetScanner, DatasetValidator
    from dataset_manager import MetadataGenerator, LabelGenerator
    from dataset_manager import StatisticsGenerator
    from dataset_manager import ImageRecord

Author: Medical AI System — Module 1
"""

from dataset_manager.scanner import DatasetScanner, ImageRecord
from dataset_manager.validator import DatasetValidator, ValidationReport, ValidationIssue
from dataset_manager.metadata_generator import MetadataGenerator
from dataset_manager.label_generator import LabelGenerator
from dataset_manager.statistics import StatisticsGenerator

__all__ = [
    # Scanner
    "DatasetScanner",
    "ImageRecord",
    # Validator
    "DatasetValidator",
    "ValidationReport",
    "ValidationIssue",
    # Generators
    "MetadataGenerator",
    "LabelGenerator",
    "StatisticsGenerator",
]

__version__ = "1.0.0"
__author__ = "Medical AI System — Module 1"
