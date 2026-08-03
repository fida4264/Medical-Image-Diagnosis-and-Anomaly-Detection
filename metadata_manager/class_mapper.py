"""
metadata_manager/class_mapper.py
==================================
Rule-based class name normalization engine.

Takes all unique raw disease class names from Module 1 and maps each to a
canonical lowercase normalized form using the rules defined in config.py.
Produces class_mapping.csv.

Design
------
Normalization is purely lookup-based (no ML). Rules are maintained in
config.CLASS_NORMALIZATION_RULES. The fallback for unrecognised classes is
'unknown'. This makes the mapping deterministic, reproducible, and auditable.

Classes
-------
ClassMapper
    Builds the raw→normalized mapping and exposes it as a DataFrame and a
    plain dict for fast per-row lookups.

Author: Medical AI System — Module 2
"""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from metadata_manager.config import (
    CLASS_NORMALIZATION_RULES,
    METADATA_OUTPUT_DIR,
    CLASS_MAPPING_CSV,
)

logger = logging.getLogger(__name__)


class ClassMapper:
    """
    Maps raw disease class names to normalized canonical names.

    Parameters
    ----------
    custom_rules : list of (list[str], str), optional
        Additional normalization rules to prepend (higher priority than
        the defaults in config.py).  Each entry is
        ([raw_variant, ...], normalized_name).

    Examples
    --------
    >>> mapper = ClassMapper()
    >>> mapper.normalize("COVID-19")
    'covid'
    >>> mapper.normalize("No Tumor")
    'notumor'
    >>> mapper.normalize("colon_aca")
    'colon_adenocarcinoma'
    """

    def __init__(
        self,
        custom_rules: Optional[List[Tuple[List[str], str]]] = None,
    ) -> None:
        all_rules = list(custom_rules or []) + list(CLASS_NORMALIZATION_RULES)
        # Build flat lookup: lowercased_raw -> normalized
        self._lookup: Dict[str, str] = {}
        for variants, normalized in all_rules:
            for v in variants:
                key = v.strip().lower()
                self._lookup[key] = normalized
        self._mapping_df: Optional[pd.DataFrame] = None
        logger.debug("ClassMapper initialised with %d lookup entries.", len(self._lookup))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, raw_class: str) -> str:
        """
        Return the normalized class name for a raw class string.

        Parameters
        ----------
        raw_class : str
            Raw class name (any casing, spacing, punctuation).

        Returns
        -------
        str
            Canonical lowercase normalized class name.
            Returns 'unknown' if no rule matches.
        """
        if not isinstance(raw_class, str):
            return "unknown"
        key = raw_class.strip().lower()
        if key in self._lookup:
            return self._lookup[key]
        # Try collapsing whitespace / underscores / hyphens
        simplified = re.sub(r"[\s_\-]+", "", key)
        for lookup_key, normalized in self._lookup.items():
            if re.sub(r"[\s_\-]+", "", lookup_key) == simplified:
                return normalized
        return "unknown"

    def build_mapping(self, raw_classes: List[str]) -> pd.DataFrame:
        """
        Build a DataFrame mapping every unique raw class to its normalized form.

        Parameters
        ----------
        raw_classes : list of str
            All unique raw class names to map.

        Returns
        -------
        pd.DataFrame
            Columns: original_class, normalized_class
        """
        unique_classes = sorted(set(str(c) for c in raw_classes))
        rows = [
            {"original_class": cls, "normalized_class": self.normalize(cls)}
            for cls in unique_classes
        ]
        self._mapping_df = pd.DataFrame(rows, columns=["original_class", "normalized_class"])
        logger.info(
            "Class mapping built: %d unique raw classes -> %d unique normalized classes.",
            len(self._mapping_df),
            self._mapping_df["normalized_class"].nunique(),
        )
        return self._mapping_df

    def apply(self, series: pd.Series) -> pd.Series:
        """
        Vectorised normalization applied to a pandas Series.

        Parameters
        ----------
        series : pd.Series
            Series of raw class name strings.

        Returns
        -------
        pd.Series
            Series of normalized class names, same index.
        """
        return series.map(self.normalize)

    def save(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save class_mapping.csv to disk.

        Parameters
        ----------
        output_dir : Path, optional
            Destination directory.  Defaults to METADATA_OUTPUT_DIR.

        Returns
        -------
        Path
            Path of the saved file.

        Raises
        ------
        RuntimeError
            If build_mapping() has not been called first.
        """
        if self._mapping_df is None:
            raise RuntimeError(
                "ClassMapper.build_mapping() must be called before save()."
            )
        dest_dir = Path(output_dir) if output_dir else METADATA_OUTPUT_DIR
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / CLASS_MAPPING_CSV
        self._mapping_df.to_csv(dest, index=False, encoding="utf-8")
        logger.info("class_mapping.csv saved -> %s  (%d rows)", dest, len(self._mapping_df))
        return dest

    @property
    def mapping_df(self) -> Optional[pd.DataFrame]:
        """The mapping DataFrame, or None if build_mapping() not yet called."""
        return self._mapping_df

    @property
    def lookup(self) -> Dict[str, str]:
        """Read-only view of the raw→normalized lookup dict."""
        return dict(self._lookup)
