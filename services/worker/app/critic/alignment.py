"""Alignment critic — pipeline node 7's alignment signal.

SigLIP text-image similarity between the panel's intended content
(`Panel.action` + a caption) and the actual generated candidate — catches a
failure mode the identity/style critics can't: a candidate that's perfectly
on-identity and on-style but shows the wrong scene entirely.

Two details that matter for a correct score: `padding="max_length"` is
required on the processor call (SigLIP was trained this way; omitting it
silently changes the score), and similarity is
`torch.sigmoid(outputs.logits_per_image)`, not a softmax — SigLIP's own
training objective is a pairwise sigmoid loss, unlike CLIP's softmax.

CPU-only; `google/siglip-base-patch16-224` is small enough for CPU inference.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

MODEL_NAME = "google/siglip-base-patch16-224"

_processor: AutoProcessor | None = None
_model: AutoModel | None = None


def _get_model() -> tuple[AutoProcessor, AutoModel]:
    """Lazy singleton, so a batch of candidates shares one loaded model
    rather than reloading per call."""
    global _processor, _model
    if _model is None:
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()  # type: ignore[union-attr]
    assert _processor is not None
    return _processor, _model


def score_alignment(image_bgr: np.ndarray, text: str) -> float:
    """Returns a `[0, 1]` similarity between `text` (typically
    `f"{panel.action} {panel.caption_a}"`) and the image, using SigLIP's own
    sigmoid scoring convention. Takes BGR, matching this project's
    convention elsewhere (`faces.py`)."""
    processor, model = _get_model()
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image_rgb)
    # Auto* instances proxy __call__ dynamically; mypy can't see that from
    # their declared types.
    inputs = processor(  # type: ignore[operator]
        text=[text], images=image, padding="max_length", return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)  # type: ignore[operator]
    logit = outputs.logits_per_image[0, 0]
    return float(torch.sigmoid(logit))
