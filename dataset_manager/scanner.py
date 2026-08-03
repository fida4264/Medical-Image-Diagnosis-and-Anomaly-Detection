"""
scanner.py
==========
Recursive dataset discovery engine for the Medical AI System.

The scanner walks the datasets directory and produces a list of ImageRecord
objects — one per image file — with zero hardcoded modality or disease names.

It handles two common dataset layouts:

    Layout A — flat disease folders:
        datasets/
            chest_xray/
                Pneumonia/
                    img001.jpg
                Normal/
                    img002.jpg

    Layout B — split sub-folders inside each disease folder:
        datasets/
            brain_mri/
                Glioma/
                    train/
                        img001.jpg
                    test/
                        img002.jpg
                No_Tumor/
                    train/ ...

    Layout C — split folders directly under the modality (no disease level):
        datasets/
            skin/
                train/
                    Melanoma/
                        img.jpg
                val/
                    Melanoma/
                        img.jpg

The scanner auto-detects which layout is in use for each modality.

Classes
-------
ImageRecord     Immutable dataclass representing a single discovered image.
DatasetScanner  Main scanner class that drives recursive discovery.

Author: Medical AI System — Module 1
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dataset_manager.config import (
    DATASETS_ROOT,
    KNOWN_MODALITIES,
    SPLIT_KEYWORDS,
    SUPPORTED_EXTENSIONS,
)
from dataset_manager.image_utils import is_hidden_file, is_supported_image

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRecord:
    """
    Immutable record describing a single medical image found on disk.

    Attributes
    ----------
    image_id : str
        Unique identifier generated from the relative path (used as primary key
        in metadata.csv, labels.csv, and all future modules).
    modality : str
        The imaging modality folder name (e.g., "brain_mri", "chest_xray").
    disease_class : str
        The disease / class label folder name (e.g., "Pneumonia", "Glioma").
        This is the raw folder name — label encoding happens in LabelGenerator.
    split : str
        Dataset split: "train", "val", "test", or "unspecified" when no split
        sub-folder was found.
    image_path : Path
        Absolute path to the image file.
    filename : str
        Bare file name including extension (e.g., "img001.jpg").
    extension : str
        Lower-cased file extension including the dot (e.g., ".jpg").
    file_size_bytes : int
        File size in bytes at scan time.
    """
    image_id: str
    modality: str
    disease_class: str
    split: str
    image_path: Path
    filename: str
    extension: str
    file_size_bytes: int


# ---------------------------------------------------------------------------
# Scanner
# ---------------------------------------------------------------------------

class DatasetScanner:
    """
    Recursively scans the datasets directory and builds a flat list of
    ImageRecord objects for every valid image found.

    Parameters
    ----------
    datasets_root : Path, optional
        Override the default datasets root path from config.

    Examples
    --------
    >>> scanner = DatasetScanner()
    >>> records = scanner.scan()
    >>> print(f"Found {len(records)} images across {scanner.modality_count} modalities")
    """

    def __init__(self, datasets_root: Optional[Path] = None) -> None:
        self._root: Path = datasets_root or DATASETS_ROOT
        self._records: List[ImageRecord] = []
        self._skipped_hidden: int = 0
        self._skipped_unsupported: int = 0
        self._discovered_modalities: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self) -> List[ImageRecord]:
        """
        Execute the full dataset scan and return all discovered ImageRecord objects.

        Returns
        -------
        list of ImageRecord
            One record per discovered image file. Order matches the filesystem
            walk order (typically alphabetical on most OS/filesystem combos).

        Raises
        ------
        FileNotFoundError
            When the datasets root directory does not exist.
        """
        if not self._root.exists():
            raise FileNotFoundError(
                f"Datasets root directory not found: {self._root}\n"
                "Please create the 'datasets/' folder and add your image data."
            )

        logger.info("Starting dataset scan from root: %s", self._root)
        self._records.clear()
        self._skipped_hidden = 0
        self._skipped_unsupported = 0
        self._discovered_modalities.clear()

        modality_dirs = self._get_modality_directories()
        logger.info("Discovered %d modality director(ies): %s",
                    len(modality_dirs), [d.name for d in modality_dirs])

        for modality_dir in modality_dirs:
            self._scan_modality(modality_dir)

        logger.info(
            "Scan complete. Total images: %d | Modalities: %d | "
            "Skipped (hidden): %d | Skipped (unsupported): %d",
            len(self._records),
            len(self._discovered_modalities),
            self._skipped_hidden,
            self._skipped_unsupported,
        )
        self._warn_on_missing_known_modalities(modality_dirs)
        return list(self._records)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def modality_count(self) -> int:
        """Number of modality directories discovered in the last scan."""
        return len(self._discovered_modalities)

    @property
    def discovered_modalities(self) -> List[str]:
        """Names of modality directories discovered in the last scan."""
        return list(self._discovered_modalities)

    @property
    def total_images(self) -> int:
        """Total number of image records produced by the last scan."""
        return len(self._records)

    @property
    def skipped_hidden(self) -> int:
        """Number of hidden files encountered and skipped."""
        return self._skipped_hidden

    @property
    def skipped_unsupported(self) -> int:
        """Number of files with unsupported extensions skipped."""
        return self._skipped_unsupported

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _get_modality_directories(self) -> List[Path]:
        """
        Return all immediate subdirectories of the datasets root.

        Hidden directories (starting with '.') are excluded.
        """
        return sorted(
            d for d in self._root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def _scan_modality(self, modality_dir: Path) -> None:
        """
        Scan a single modality directory and append records to self._records.

        Detects the layout in use (A, B, or C) and dispatches accordingly.
        """
        modality_name = modality_dir.name
        self._discovered_modalities.append(modality_name)
        logger.debug("Scanning modality: %s", modality_name)

        # Check whether immediate children contain split keywords → Layout C
        immediate_children = [
            d for d in modality_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

        if self._all_are_splits(immediate_children):
            # Layout C: datasets/modality/split/disease/images
            logger.debug("  Layout C (split → disease) detected for '%s'", modality_name)
            self._scan_layout_c(modality_dir, modality_name)
        else:
            # Layout A or B: datasets/modality/disease/[split/]images
            logger.debug("  Layout A/B (disease → [split →]) detected for '%s'", modality_name)
            self._scan_layout_ab(modality_dir, modality_name)

    def _scan_layout_ab(self, modality_dir: Path, modality_name: str) -> None:
        """
        Handle modality directories where disease folders are the first level.

        Under each disease folder, there may be either:
        - Direct image files     → Layout A (split="unspecified")
        - Split sub-folders      → Layout B (split="train"/"val"/"test")
        """
        disease_dirs = sorted(
            d for d in modality_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for disease_dir in disease_dirs:
            disease_name = disease_dir.name
            immediate = [
                d for d in disease_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            if self._all_are_splits(immediate):
                # Layout B — split folders exist under disease
                for split_dir in sorted(immediate):
                    split_name = self._normalise_split(split_dir.name)
                    self._collect_images(split_dir, modality_name, disease_name, split_name)
            else:
                # Layout A — images directly under disease folder
                self._collect_images(disease_dir, modality_name, disease_name, "unspecified")

    def _scan_layout_c(self, modality_dir: Path, modality_name: str) -> None:
        """
        Handle modality directories where split folders are the first level.

        Structure: datasets/modality/split/disease/images
        """
        split_dirs = sorted(
            d for d in modality_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

        for split_dir in split_dirs:
            split_name = self._normalise_split(split_dir.name)
            disease_dirs = sorted(
                d for d in split_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            )
            for disease_dir in disease_dirs:
                disease_name = disease_dir.name
                self._collect_images(disease_dir, modality_name, disease_name, split_name)

    def _collect_images(
        self,
        folder: Path,
        modality: str,
        disease_class: str,
        split: str,
    ) -> None:
        """
        Walk *folder* recursively and create an ImageRecord for every
        supported, non-hidden image file found.
        """
        for file_path in sorted(folder.rglob("*")):
            if not file_path.is_file():
                continue

            # Skip hidden files / files inside hidden directories
            try:
                relative = file_path.relative_to(self._root)
            except ValueError:
                relative = file_path

            if is_hidden_file(relative):
                self._skipped_hidden += 1
                logger.debug("Skipping hidden file: %s", file_path)
                continue

            # Skip unsupported extensions
            if not is_supported_image(file_path):
                self._skipped_unsupported += 1
                logger.debug("Skipping unsupported file: %s", file_path)
                continue

            # Skip mask files (e.g. segmentation ground truth in ultrasound)
            if "_mask" in file_path.stem.lower():
                logger.debug("Skipping mask file: %s", file_path)
                continue

            # Build a stable, unique image_id from the relative path
            image_id = str(relative).replace("\\", "/")

            try:
                file_size = file_path.stat().st_size
            except OSError:
                file_size = 0

            record = ImageRecord(
                image_id=image_id,
                modality=modality,
                disease_class=disease_class,
                split=split,
                image_path=file_path.resolve(),
                filename=file_path.name,
                extension=file_path.suffix.lower(),
                file_size_bytes=file_size,
            )
            self._records.append(record)
            logger.debug("  + %s", image_id)

    # ------------------------------------------------------------------
    # Layout-Detection Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_are_splits(dirs: List[Path]) -> bool:
        """
        Return True if every directory in *dirs* has a name matching a known
        split keyword (case-insensitive).

        An empty list returns False (we cannot call an empty folder 'all splits').
        """
        if not dirs:
            return False
        return all(d.name.lower() in SPLIT_KEYWORDS for d in dirs)

    @staticmethod
    def _normalise_split(name: str) -> str:
        """
        Map variant split names to a canonical form.

        Maps:
            "training"   → "train"
            "valid"      → "val"
            "validation" → "val"
            "testing"    → "test"
            anything else → name.lower()
        """
        mapping = {
            "training": "train",
            "valid": "val",
            "validation": "val",
            "testing": "test",
        }
        return mapping.get(name.lower(), name.lower())

    def _warn_on_missing_known_modalities(self, found_dirs: List[Path]) -> None:
        """
        Log a warning for every entry in KNOWN_MODALITIES that was not found.
        """
        found_names = {d.name.lower() for d in found_dirs}
        for expected in KNOWN_MODALITIES:
            if expected.lower() not in found_names:
                logger.warning(
                    "Expected modality folder '%s' was NOT found in '%s'. "
                    "This is fine if you haven't downloaded that dataset yet.",
                    expected,
                    self._root,
                )
