"""
unknown_configuration.py — Module 16.4: Unknown Disease Detection
Medical AI Platform — Configuration dataclass for the Unknown Disease Detection subsystem.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict
from .unknown_exceptions import UnknownConfigurationError


@dataclass
class UnknownConfiguration:
    """
    Configuration for the Unknown Disease Detection subsystem.

    Attributes
    ----------
    novelty_threshold : float
        Score above which a sample is considered an unknown disease (0.0–1.0).
    similarity_threshold : float
        Minimum cosine similarity to nearest prototype to be considered KNOWN.
    embedding_dim : int
        Dimensionality of feature embedding vectors.
    min_samples_per_class : int
        Minimum training samples required to build a class prototype.
    use_cosine_similarity : bool
        Use cosine similarity (True) or Euclidean distance (False) for prototype matching.
    per_modality_thresholds : Dict[str, float]
        Optional per-modality novelty threshold overrides.
    """
    novelty_threshold: float = 0.60
    similarity_threshold: float = 0.70
    embedding_dim: int = 512
    min_samples_per_class: int = 5
    use_cosine_similarity: bool = True
    per_modality_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "brain_mri": 0.58,
            "chest_xray": 0.60,
            "brain_ct": 0.58,
            "histopathology": 0.55,
            "retina": 0.60,
            "skin": 0.62,
            "ultrasound": 0.65,
        }
    )

    def __post_init__(self) -> None:
        if not 0.0 < self.novelty_threshold < 1.0:
            raise UnknownConfigurationError(
                f"novelty_threshold must be in (0, 1); got {self.novelty_threshold}"
            )
        if not 0.0 < self.similarity_threshold < 1.0:
            raise UnknownConfigurationError(
                f"similarity_threshold must be in (0, 1); got {self.similarity_threshold}"
            )
        if self.embedding_dim <= 0:
            raise UnknownConfigurationError(f"embedding_dim must be > 0; got {self.embedding_dim}")

    def get_novelty_threshold(self, modality: str) -> float:
        return self.per_modality_thresholds.get(modality, self.novelty_threshold)

    @classmethod
    def default(cls) -> "UnknownConfiguration":
        return cls()

    @classmethod
    def fast_test(cls) -> "UnknownConfiguration":
        return cls(embedding_dim=64, min_samples_per_class=3)


__all__ = ["UnknownConfiguration"]
