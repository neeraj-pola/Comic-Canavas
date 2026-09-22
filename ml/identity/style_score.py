"""DINOv2-small style scoring — the reference-bank similarity used to rank
master candidates and score panel style.

CPU-only: `facebook/dinov2-small` is a 22M-parameter ViT-S/14, one forward
pass per image runs in well under a second on CPU. `AutoModel(**AutoImage
Processor(...))` gives `.pooler_output` directly as the (1, 384) CLS
embedding, no manual slicing of `last_hidden_state` needed.
`AutoImageProcessor` requires `torchvision` importable at all (an
`ImportError`, not a lazily-used optional backend), hence
`ml/pyproject.toml` lists it explicitly even though this module never
imports it directly.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from ml.identity.faces import decode_jpeg_bgr, detect_and_embed

MODEL_NAME = "facebook/dinov2-small"
_REPO_ROOT = Path(__file__).resolve().parents[2]
REG_ROOT = _REPO_ROOT / "ml" / "reg" / "comic_character"
DEFAULT_BANK_PATH = (
    _REPO_ROOT / "ml" / "identity" / "reg_embeddings" / "comic_character_dinov2_small.npy"
)
DEFAULT_TOP_K = 10

_processor: AutoImageProcessor | None = None
_model: AutoModel | None = None


class EmptyReferenceSetError(ValueError):
    def __init__(self, image_dir: Path) -> None:
        super().__init__(f"no images found under {image_dir} to build a reference bank from")


def _get_model() -> tuple[AutoImageProcessor, AutoModel]:
    """Lazy singleton — same reasoning as `faces.py`'s `_get_app`: a batch
    of images should share one loaded model, not reload per call."""
    global _processor, _model
    if _model is None:
        _processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()  # type: ignore[union-attr]  # Auto* dynamic-dispatch classes, see module docstring
    assert _processor is not None
    return _processor, _model


def embed_image(image_bgr: np.ndarray) -> np.ndarray:
    """(384,) float32, L2-normalized CLS embedding. BGR in, matching this
    project's convention everywhere else (`faces.py`) — converted to RGB
    here since that's what the image processor expects."""
    processor, model = _get_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    # Auto* instances proxy __call__ dynamically; mypy can't see that from
    # their declared types (see module docstring) — type: ignore, not a
    # real typing gap in our own code.
    inputs = processor(images=image_rgb, return_tensors="pt")  # type: ignore[operator]
    with torch.no_grad():
        outputs = model(**inputs)  # type: ignore[operator]
    embedding = outputs.pooler_output[0].numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        raise ValueError("embed_image: model produced a zero vector")
    return embedding / norm


def embed_jpeg(data: bytes) -> np.ndarray:
    return embed_image(decode_jpeg_bgr(data))


def build_reference_bank(
    image_dir: Path = REG_ROOT, out_path: Path = DEFAULT_BANK_PATH
) -> np.ndarray:
    """One-time setup: embeds every `*.jpg` under `image_dir` into an
    `(N, 384)` array, saved to `out_path`. Raises if the directory has no
    images — an empty bank would silently make every candidate's style
    score meaningless rather than failing loudly."""
    paths = sorted(image_dir.glob("*.jpg"))
    if not paths:
        raise EmptyReferenceSetError(image_dir)
    embeddings = np.stack([embed_jpeg(p.read_bytes()) for p in paths])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, embeddings)
    return embeddings


def load_reference_bank(path: Path = DEFAULT_BANK_PATH) -> np.ndarray:
    return np.load(path)


def cosine_to(embedding: np.ndarray, bank: np.ndarray) -> np.ndarray:
    """(N,) cosines — rows of `bank` are already unit-normalized (both
    `build_reference_bank` and `embed_image` normalize), so this is one
    matmul, not N separate divisions."""
    return bank @ embedding


def master_face_embedding(master_bgr: np.ndarray) -> np.ndarray | None:
    """DINOv2 embedding of just the master's face crop — the fixed
    reference point `identity_similarity` compares every candidate
    against. Returns `None` if no face is detected in the master itself
    (shouldn't happen for an approved one, but callers should handle it
    rather than crash on an unexpected input)."""
    detected = detect_and_embed(master_bgr)
    if detected is None:
        return None
    return embed_image(detected.crop_bgr)


def identity_similarity(candidate_bgr: np.ndarray, master_embedding: np.ndarray) -> float | None:
    """DINOv2 cosine similarity between a candidate's face crop and the
    master's face crop — deliberately not a whole-frame comparison: a
    tight headshot (`master.png`) and a medium/wide daily panel differ
    enormously in composition and background regardless of identity, and
    a whole-image embedding would pick up on that composition difference
    rather than just the face. `style_score`/`cosine_to` are a separate,
    unrelated comparison (against the many-subject regularization bank),
    not a replacement for this.

    Returns `None` if no face is detected in the candidate — the same
    InsightFace/RetinaFace limitation on stylized art documented in
    `master.py`'s `rank_candidates`; callers should treat that as "signal
    unavailable," not a hard failure."""
    detected = detect_and_embed(candidate_bgr)
    if detected is None:
        return None
    candidate_embedding = embed_image(detected.crop_bgr)
    return float(cosine_to(candidate_embedding, master_embedding.reshape(1, -1))[0])


def within_sheet_similarity(embeddings: list[np.ndarray]) -> float:
    """Mean pairwise cosine similarity among a set of already face-cropped,
    unit-normalized DINOv2 embeddings — distinct from `identity_similarity`
    's one-candidate-to-master comparison: this measures whether the kept
    sheet reads as one consistent character among itself, not individually
    against the master. `1.0` for fewer than 2 embeddings (nothing to
    compare)."""
    if len(embeddings) < 2:
        return 1.0
    stacked = np.stack(embeddings)
    similarities = stacked @ stacked.T
    n = len(embeddings)
    off_diagonal_sum = similarities.sum() - n  # each self-similarity is exactly 1.0
    return float(off_diagonal_sum / (n * (n - 1)))


def style_score(
    image_bgr: np.ndarray, reference: np.ndarray, *, top_k: int = DEFAULT_TOP_K
) -> float:
    """Mean of the top-k cosines to the reference bank, not the mean over
    all N: the regularization set spans many subjects/compositions, so a
    candidate only needs to read as on-style compared to *some* of them,
    not all — mean-over-all would compress every candidate into a narrow
    band and destroy the ranking signal this exists to provide."""
    return style_score_from_embedding(embed_image(image_bgr), reference, top_k=top_k)


def style_score_from_embedding(
    embedding: np.ndarray, reference: np.ndarray, *, top_k: int = DEFAULT_TOP_K
) -> float:
    """`style_score` for an image that was already embedded (the scorer embeds each candidate
    once and reuses the vector for the style signal and for the ranking model)."""
    cosines = cosine_to(embedding, reference)
    k = min(top_k, len(cosines))
    top = np.sort(cosines)[-k:]
    return float(np.mean(top))
