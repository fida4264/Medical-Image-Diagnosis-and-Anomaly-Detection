"""
label_generator.py
==================
Generates integer label mappings and class distribution statistics.

Produces two output files:

labels.csv
----------
One row per image. Maps every image_id to an integer label and its human-readable
class name. Labels are assigned per modality (each modality has its own
0-based integer namespace) so that the same folder name in different modalities
does not collide.

Columns:
    image_id        Primary key matching metadata.csv.
    modality        Imaging modality.
    disease_class   Raw class name (folder name).
    split           Dataset split.
    label           Zero-based integer label within the modality.
    class_name      Canonical class name (same as disease_class, normalised).

class_distribution.csv
-----------------------
Aggregated counts of images per (modality, disease_class, split) combination.

Columns:
    modality        Imaging modality.
    disease_class   Disease / class name.
    split           Dataset split.
    count           Number of images in that (modality, class, split) cell.

Classes
-------
LabelGenerator   Builds labels.csv and class_distribution.csv.

Author: Medical AI System — Module 1
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from dataset_manager.config import (
    CLASS_DISTRIBUTION_CSV,
    COL_CLASS_NAME,
    COL_COUNT,
    COL_DISEASE_CLASS,
    COL_IMAGE_ID,
    COL_IS_VALID,
    COL_LABEL,
    COL_MODALITY,
    COL_SPLIT,
    LABELS_CSV,
    OUTPUTS_DIR,
)

logger = logging.getLogger(__name__)


class LabelGenerator:
    """
    Derives integer labels and class distributions from the metadata DataFrame.

    Parameters
    ----------
    metadata_df : pd.DataFrame
        The DataFrame produced by MetadataGenerator.generate().
    outputs_dir : Path, optional
        Directory where output CSV files will be written.

    Examples
    --------
    >>> gen = LabelGenerator(metadata_df)
    >>> labels_df, dist_df = gen.generate()
    >>> gen.save_labels()
    >>> gen.save_class_distribution()
    """

    def __init__(
        self,
        metadata_df: pd.DataFrame,
        outputs_dir: Optional[Path] = None,
    ) -> None:
        self._meta = metadata_df.copy()
        self._outputs_dir = outputs_dir or OUTPUTS_DIR
        self._labels_df: Optional[pd.DataFrame] = None
        self._dist_df: Optional[pd.DataFrame] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute label assignments and class distribution statistics.

        Returns
        -------
        tuple of (pd.DataFrame, pd.DataFrame)
            (labels_df, class_distribution_df)
        """
        logger.info("Generating label mappings and class distribution …")
        label_map = self._build_label_map()
        self._labels_df = self._build_labels_dataframe(label_map)
        self._dist_df = self._build_distribution_dataframe()
        logger.info(
            "Label generation complete. "
            "Labels shape: %s | Distribution shape: %s",
            self._labels_df.shape,
            self._dist_df.shape,
        )
        return self._labels_df, self._dist_df

    def save_labels(self, output_path: Optional[Path] = None) -> Path:
        """
        Write labels.csv to disk.

        Parameters
        ----------
        output_path : Path, optional
            Custom destination; defaults to OUTPUTS_DIR/labels.csv.

        Returns
        -------
        Path
            The path where labels.csv was written.

        Raises
        ------
        RuntimeError
            When called before generate().
        """
        self._require_generated()
        dest = output_path or (self._outputs_dir / LABELS_CSV)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._labels_df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("labels.csv saved → %s  (%d rows)", dest, len(self._labels_df))
        return dest

    def save_class_distribution(self, output_path: Optional[Path] = None) -> Path:
        """
        Write class_distribution.csv to disk.

        Parameters
        ----------
        output_path : Path, optional
            Custom destination; defaults to OUTPUTS_DIR/class_distribution.csv.

        Returns
        -------
        Path
            The path where class_distribution.csv was written.

        Raises
        ------
        RuntimeError
            When called before generate().
        """
        self._require_generated()
        dest = output_path or (self._outputs_dir / CLASS_DISTRIBUTION_CSV)
        dest.parent.mkdir(parents=True, exist_ok=True)
        self._dist_df.to_csv(dest, index=False, encoding="utf-8")
        logger.info(
            "class_distribution.csv saved → %s  (%d rows)",
            dest,
            len(self._dist_df),
        )
        return dest

    @property
    def label_map(self) -> Dict[str, Dict[str, int]]:
        """
        Return the {modality: {disease_class: integer_label}} mapping.

        Only available after calling generate().
        """
        self._require_generated()
        return self._build_label_map()

    @property
    def labels_dataframe(self) -> Optional[pd.DataFrame]:
        """The labels DataFrame, or None if generate() not called."""
        return self._labels_df

    @property
    def distribution_dataframe(self) -> Optional[pd.DataFrame]:
        """The class distribution DataFrame, or None if generate() not called."""
        return self._dist_df

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_label_map(self) -> Dict[str, Dict[str, int]]:
        """
        Build a per-modality label map: {modality: {class_name: int_label}}.

        Labels are assigned alphabetically within each modality so that the
        mapping is deterministic and reproducible regardless of scan order.
        """
        label_map: Dict[str, Dict[str, int]] = {}

        for modality in sorted(self._meta[COL_MODALITY].unique()):
            classes = sorted(
                self._meta.loc[
                    self._meta[COL_MODALITY] == modality, COL_DISEASE_CLASS
                ].unique()
            )
            label_map[modality] = {cls: idx for idx, cls in enumerate(classes)}

        return label_map

    def _build_labels_dataframe(
        self,
        label_map: Dict[str, Dict[str, int]],
    ) -> pd.DataFrame:
        """
        Construct the labels DataFrame by joining label integers into the
        metadata table.
        """
        rows = []
        for _, row in self._meta.iterrows():
            modality = row[COL_MODALITY]
            disease_class = row[COL_DISEASE_CLASS]
            label = label_map.get(modality, {}).get(disease_class, -1)
            rows.append({
                COL_IMAGE_ID:      row[COL_IMAGE_ID],
                COL_MODALITY:      modality,
                COL_DISEASE_CLASS: disease_class,
                COL_SPLIT:         row[COL_SPLIT],
                COL_LABEL:         label,
                COL_CLASS_NAME:    disease_class,
            })

        df = pd.DataFrame(rows, columns=[
            COL_IMAGE_ID,
            COL_MODALITY,
            COL_DISEASE_CLASS,
            COL_SPLIT,
            COL_LABEL,
            COL_CLASS_NAME,
        ])
        return df

    def _build_distribution_dataframe(self) -> pd.DataFrame:
        """
        Aggregate image counts per (modality, disease_class, split).

        Only valid images (is_valid == True) are counted if the column exists;
        otherwise all records are counted.
        """
        working = self._meta.copy()
        if COL_IS_VALID in working.columns:
            working = working[working[COL_IS_VALID].astype(bool)]

        dist = (
            working.groupby([COL_MODALITY, COL_DISEASE_CLASS, COL_SPLIT])
            .size()
            .reset_index(name=COL_COUNT)
        )
        dist = dist.sort_values(
            [COL_MODALITY, COL_DISEASE_CLASS, COL_SPLIT]
        ).reset_index(drop=True)
        return dist

    def _require_generated(self) -> None:
        """Raise RuntimeError if generate() has not been called."""
        if self._labels_df is None or self._dist_df is None:
            raise RuntimeError("Call generate() before accessing results or saving.")
