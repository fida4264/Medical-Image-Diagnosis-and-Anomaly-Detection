"""
metadata_generator.py
=====================
Generates the primary metadata table (metadata.csv) for the Medical AI System.

The metadata table is the single source of truth for all image records in the
project.  Every future module — preprocessing, augmentation, EDA, training,
evaluation — should load metadata.csv rather than scanning the filesystem again.

Schema of metadata.csv
-----------------------
image_id            Unique record identifier (relative path, forward-slashed).
modality            Imaging modality folder name.
disease_class       Disease / class folder name (raw, un-encoded).
split               "train" | "val" | "test" | "unspecified".
image_path          Absolute path to the image file.
filename            Bare filename including extension.
extension           Lower-cased extension including leading dot.
width               Image width in pixels (0 if unreadable).
height              Image height in pixels (0 if unreadable).
file_size_bytes     File size in bytes.
is_valid            True when the image passes the corruption check.
md5_hash            Hex MD5 digest (for deduplication by downstream modules).

Classes
-------
MetadataGenerator   Builds and saves metadata.csv from a list of ImageRecords.

Author: Medical AI System — Module 1
"""

import logging
from pathlib import Path
from typing import List, Optional

import pandas as pd

from dataset_manager.config import (
    COL_DISEASE_CLASS,
    COL_EXTENSION,
    COL_FILE_SIZE_BYTES,
    COL_FILENAME,
    COL_HEIGHT,
    COL_IMAGE_ID,
    COL_IMAGE_PATH,
    COL_IS_VALID,
    COL_MD5_HASH,
    COL_MODALITY,
    COL_SPLIT,
    COL_WIDTH,
    METADATA_CSV,
    OUTPUTS_DIR,
)
from dataset_manager.image_utils import (
    compute_md5_hash,
    get_image_dimensions,
    is_corrupted,
)
from dataset_manager.scanner import ImageRecord

logger = logging.getLogger(__name__)


class MetadataGenerator:
    """
    Converts a list of ImageRecord objects into a rich metadata DataFrame and
    persists it as metadata.csv.

    The generator enriches each record with per-image attributes that require
    actually opening the file (dimensions, MD5, validity), in addition to the
    fields already captured during scanning.

    Parameters
    ----------
    records : list of ImageRecord
        Image records from DatasetScanner.
    outputs_dir : Path, optional
        Directory where metadata.csv will be written.

    Examples
    --------
    >>> gen = MetadataGenerator(records)
    >>> df = gen.generate()
    >>> gen.save()
    >>> print(df.head())
    """

    def __init__(
        self,
        records: List[ImageRecord],
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._records = records
        self._outputs_dir = outputs_dir or OUTPUTS_DIR
        self._df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """
        Build the metadata DataFrame by enriching each ImageRecord.

        Returns
        -------
        pd.DataFrame
            One row per image with all metadata columns populated.
        """
        logger.info(
            "Generating metadata for %d image records …", len(self._records)
        )
        rows = []
        for idx, rec in enumerate(self._records, start=1):
            if idx % 500 == 0:
                logger.debug("  Processed %d / %d records …", idx, len(self._records))

            # Collect image-level attributes (requires file I/O)
            dims = get_image_dimensions(rec.image_path)
            width, height = dims if dims else (0, 0)
            valid = not is_corrupted(rec.image_path)
            md5 = compute_md5_hash(rec.image_path)

            rows.append({
                COL_IMAGE_ID:        rec.image_id,
                COL_MODALITY:        rec.modality,
                COL_DISEASE_CLASS:   rec.disease_class,
                COL_SPLIT:           rec.split,
                COL_IMAGE_PATH:      str(rec.image_path),
                COL_FILENAME:        rec.filename,
                COL_EXTENSION:       rec.extension,
                COL_WIDTH:           width,
                COL_HEIGHT:          height,
                COL_FILE_SIZE_BYTES: rec.file_size_bytes,
                COL_IS_VALID:        valid,
                COL_MD5_HASH:        md5,
            })

        self._df = pd.DataFrame(rows, columns=self._column_order())
        logger.info(
            "Metadata generation complete. Shape: %s", self._df.shape
        )
        return self._df

    def save(self, output_path: Optional[Path] = None) -> Path:
        """
        Write the metadata DataFrame to CSV.

        Parameters
        ----------
        output_path : Path, optional
            Custom destination path; defaults to OUTPUTS_DIR/metadata.csv.

        Returns
        -------
        Path
            The path where the file was written.

        Raises
        ------
        RuntimeError
            When called before generate().
        """
        if self._df is None:
            raise RuntimeError(
                "Call generate() before save()."
            )
        dest = output_path or (self._outputs_dir / METADATA_CSV)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("metadata.csv saved → %s  (%d rows)", dest, len(self._df))
        return dest

    @property
    def dataframe(self) -> Optional[pd.DataFrame]:
        """The generated metadata DataFrame, or None if generate() not called."""
        return self._df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _column_order() -> List[str]:
        """Return the canonical column ordering for metadata.csv."""
        return [
            COL_IMAGE_ID,
            COL_MODALITY,
            COL_DISEASE_CLASS,
            COL_SPLIT,
            COL_IMAGE_PATH,
            COL_FILENAME,
            COL_EXTENSION,
            COL_WIDTH,
            COL_HEIGHT,
            COL_FILE_SIZE_BYTES,
            COL_IS_VALID,
            COL_MD5_HASH,
        ]
