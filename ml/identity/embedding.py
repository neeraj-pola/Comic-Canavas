"""ArcFace reference embedding: averages several usable photos' embeddings
into one reference identity vector, and measures how well each individual
photo agrees with that average.

`IdentityModel` and its storage live in `ml/identity/model.py`.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from ml.identity.model import IdentityModel


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("cannot normalize a zero vector")
    return cast(np.ndarray, vector / norm)


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """L2-normalizing both before and after averaging keeps the result a
    valid unit vector for cosine similarity, and stops any one photo's
    raw embedding magnitude from skewing the average."""
    normalized = [_normalize(e) for e in embeddings]
    mean = np.mean(normalized, axis=0)
    return _normalize(mean)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


class NoEmbeddingsError(ValueError):
    def __init__(self) -> None:
        super().__init__("build_identity_model: no embeddings given")


def build_identity_model(person_id: str, embeddings: list[np.ndarray]) -> IdentityModel:
    if not embeddings:
        raise NoEmbeddingsError
    reference = average_embedding(embeddings)
    similarities = [cosine_similarity(e, reference) for e in embeddings]
    return IdentityModel(
        person_id=person_id,
        embedding=reference.tolist(),
        source_photo_count=len(embeddings),
        mean_self_cosine=float(np.mean(similarities)),
    )
