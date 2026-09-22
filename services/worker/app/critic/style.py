"""Style critic — pipeline node 7's style signal.

Wraps `ml/identity/style_score.py`'s DINOv2-small embedding + top-k cosine
similarity against the shared regularization-set bank, used as the style
reference set.
"""

from __future__ import annotations

import numpy as np

from ml.identity.faces import decode_jpeg_bgr
from ml.identity.style_score import (
    DEFAULT_TOP_K,
    load_reference_bank,
    style_score,
    style_score_from_embedding,
)


def score_style(
    image_bytes: bytes, *, reference_bank: np.ndarray | None = None, top_k: int = DEFAULT_TOP_K
) -> float:
    """Returns a style-match score — mean of the top-`k` cosine similarities
    to the reference bank, so a candidate only needs to read as on-style
    compared to *some* of the many-subject bank, not all of it.
    `reference_bank` defaults to the committed bank; callers pass their own
    for a different reference set."""
    bank = reference_bank if reference_bank is not None else load_reference_bank()
    image_bgr = decode_jpeg_bgr(image_bytes)
    return style_score(image_bgr, bank, top_k=top_k)


def score_style_from_embedding(
    embedding: np.ndarray, *, reference_bank: np.ndarray | None = None, top_k: int = DEFAULT_TOP_K
) -> float:
    """`score_style` for an image whose DINOv2 embedding is already in hand."""
    bank = reference_bank if reference_bank is not None else load_reference_bank()
    return style_score_from_embedding(embedding, bank, top_k=top_k)
