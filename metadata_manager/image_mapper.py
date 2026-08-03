"""
metadata_manager/image_mapper.py
==================================
Generates unique internal image IDs and produces image_mapping.csv.

Each image receives a globally unique ID in the format:
    {MODALITY_PREFIX}_{SEQUENCE:06d}
    e.g. MRI_000001, XRAY_000042, HISTO_025000

IDs are sequential within each modality and guaranteed unique across the
entire dataset.  The mapping is stored in image_mapping.csv so any
downstream module can always reconstruct the original path from an ID.

Classes
-------
ImageMapper
    Generates internal IDs and builds the image_mapping DataFrame.

Author: Medical AI System — Module 2
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from metadata_manager.config import (
    MODALITY_ABBREVIATION,
    METADATA_OUTPUT_DIR,
    IMAGE_MAPPING_CSV,
    COL_IMAGE_ID,
    COL_INTERNAL_ID,
    COL_MODALITY,
    COL_IMAGE_PATH,
)

logger = logging.getLogger(__name__)


class ImageMapper:
    """
    Assigns unique internal IDs to every image and builds image_mapping.csv.

    Parameters
    ----------
    master_df : pd.DataFrame
        The standardised master_metadata DataFrame from MetadataStandardizer.
    outputs_dir : Path, optional
        Directory where image_mapping.csv will be saved.

    Usage
    -----
        mapper = ImageMapper(master_df)
        id_series = mapper.generate_ids()
        mapper.save()
    """

    def __init__(
        self,
        master_df: pd.DataFrame,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._master_df: pd.DataFrame = master_df
        self._outputs_dir: Path = Path(outputs_dir) if outputs_dir else METADATA_OUTPUT_DIR
        self._mapping_df: Optional[pd.DataFrame] = None
        self._id_series: Optional[pd.Series] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_ids(self) -> pd.Series:
        """
        Generate sequential internal IDs for every row in master_df.

        ID format: {PREFIX}_{N:06d}
        Prefix is derived from MODALITY_ABBREVIATION in config.py.
        Sequence resets to 1 for each modality.

        Returns
        -------
        pd.Series
            Series of internal ID strings, indexed identically to master_df.

        Examples
        --------
        brain_mri rows  → MRI_000001, MRI_000002, ...
        chest_xray rows → XRAY_000001, XRAY_000002, ...
        """
        logger.info("Generating internal image IDs ...")
        df = self._master_df.copy()

        # Counter per modality prefix
        counters: dict = {}
        ids = []

        for modality in df[COL_MODALITY]:
            prefix = MODALITY_ABBREVIATION.get(str(modality), "IMG")
            counters[prefix] = counters.get(prefix, 0) + 1
            ids.append(f"{prefix}_{counters[prefix]:06d}")

        self._id_series = pd.Series(ids, index=df.index, name=COL_INTERNAL_ID)

        # Build mapping DataFrame — all operations are O(n) vectorised
        extensions = df["extension"] if "extension" in df.columns else pd.Series([""] * len(df), index=df.index)
        self._mapping_df = pd.DataFrame({
            COL_INTERNAL_ID: self._id_series,
            COL_IMAGE_ID: df[COL_IMAGE_ID],
            "original_filename": df[COL_IMAGE_PATH].map(lambda p: Path(p).name),
            "original_path": df[COL_IMAGE_PATH],
            "new_internal_name": self._id_series.values + extensions.values,
            COL_MODALITY: df[COL_MODALITY],
        }, index=df.index)

        logger.info(
            "Internal IDs generated: %d total across %d modalities.",
            len(ids),
            len(counters),
        )
        return self._id_series

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save image_mapping.csv to disk.

        Returns
        -------
        Path
            Path of the saved file.

        Raises
        ------
        RuntimeError
            If generate_ids() has not been called first.
        """
        if self._mapping_df is None:
            raise RuntimeError("generate_ids() must be called before save().")
        dest_dir = Path(output_dir) if output_dir else self._outputs_dir
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / IMAGE_MAPPING_CSV
        self._mapping_df.to_csv(dest, index=False, encoding="utf-8")
        logger.info(
            "image_mapping.csv saved -> %s  (%d rows)", dest, len(self._mapping_df)
        )
        return dest

    @property
    def mapping_df(self) -> Optional[pd.DataFrame]:
        """The image mapping DataFrame, or None before generate_ids()."""
        return self._mapping_df

    @property
    def id_series(self) -> Optional[pd.Series]:
        """The internal ID Series, or None before generate_ids()."""
        return self._id_series

    def lookup_original(self, internal_id: str) -> Optional[str]:
        """
        Return the original image path for a given internal ID.

        Parameters
        ----------
        internal_id : str
            e.g. 'MRI_000042'

        Returns
        -------
        str or None
            Original absolute image path, or None if not found.
        """
        if self._mapping_df is None:
            return None
        rows = self._mapping_df[self._mapping_df[COL_INTERNAL_ID] == internal_id]
        if rows.empty:
            return None
        return rows.iloc[0]["original_path"]
