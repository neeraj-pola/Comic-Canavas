"""The image embedding as ranking features.

Each scored candidate keeps its DINOv2-small embedding (384 numbers). Too many to learn from a
few dozen picks, so the ranking model sees only its first few principal components — the
directions along which this person's candidate images differ most (pose, framing, composition,
expression) — standardised, with a tight prior so a component earns a weight only when the
picks insist. The projection is fitted on the person's own candidates at every rebuild and
stored next to the model it belongs to, so a model is only ever applied with the projection
it was trained with.

Included because it is cheap and held-out-gated like everything else, not because it has been
shown to beat the named features on a small sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from app.preference.features import EMBED_DIMS

MIN_VECTORS = 12  # fewer embeddings than this: no projection (the components read as zero)


@dataclass
class Projection:
    mean: np.ndarray
    components: np.ndarray  # (EMBED_DIMS, d)
    std: np.ndarray  # (EMBED_DIMS,)

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": [round(float(x), 5) for x in self.mean],
            "components": [[round(float(x), 5) for x in row] for row in self.components],
            "std": [round(float(x), 5) for x in self.std],
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> Projection | None:
        if not data or "mean" not in data:
            return None
        return Projection(
            mean=np.array(data["mean"], dtype=np.float64),
            components=np.array(data["components"], dtype=np.float64),
            std=np.array(data["std"], dtype=np.float64),
        )


def fit_projection(vectors: list[np.ndarray]) -> Projection | None:
    """PCA of the given embeddings, or `None` when there are too few to mean anything."""
    if len(vectors) < MIN_VECTORS:
        return None
    matrix = np.stack(vectors).astype(np.float64)
    mean = matrix.mean(axis=0)
    _, _, vt = np.linalg.svd(matrix - mean, full_matrices=False)
    components = vt[:EMBED_DIMS]
    # A fixed sign for each component (its largest entry positive), so the same data always
    # gives the same projection.
    for i in range(len(components)):
        if components[i][np.argmax(np.abs(components[i]))] < 0:
            components[i] = -components[i]
    projected = (matrix - mean) @ components.T
    std = projected.std(axis=0)
    std = np.where(std > 1e-9, std, 1.0)
    if len(components) < EMBED_DIMS:  # fewer vectors than components: pad with dead directions
        pad = EMBED_DIMS - len(components)
        components = np.vstack([components, np.zeros((pad, components.shape[1]))])
        std = np.concatenate([std, np.ones(pad)])
    return Projection(mean=mean, components=components, std=std)


def project(
    projection: Projection | None, embedding: list[float] | np.ndarray | None
) -> np.ndarray:
    """The `EMBED_DIMS` standardised components of one embedding; zeros when there is no
    projection or no embedding (a candidate that reads as "average" on all of them)."""
    if projection is None or embedding is None:
        return np.zeros(EMBED_DIMS)
    vector = np.asarray(embedding, dtype=np.float64)
    if vector.shape != projection.mean.shape:
        return np.zeros(EMBED_DIMS)
    projected: np.ndarray = ((vector - projection.mean) @ projection.components.T) / projection.std
    return projected
