"""Identity critic — pipeline node 7's identity signal.

Checklist-primary, weak-ArcFace-secondary: ArcFace is built for photographic
face recognition and frequently can't detect a face in stylized comic art, so
the VLM feature checklist (does this candidate show every feature of the
approved character) is the dominant signal, with ArcFace agreement as a weak
tie-break. The same weighting `ml/identity/master.py`/`sheet.py` use for
ranking master/sheet candidates, reused here for per-panel candidates.
"""

from __future__ import annotations

import numpy as np

from contracts import LookCard
from ml.identity.checklist import score_checklist
from ml.identity.embedding import cosine_similarity
from ml.identity.faces import (
    decode_jpeg_bgr,
    detect_and_embed,
    detect_face_box,
    encode_crop_jpeg,
)

W_CHECKLIST = 0.7  # dominant signal — the checklist is the real identity gate
W_ARCFACE = 0.3  # weak secondary/tie-break signal only

# On a small, distant face, InsightFace still finds a face and the checklist
# is handed a heavily upscaled crop — coarse cues can look confidently
# correct even when the face is too low-detail to verify identity. Below
# `FACE_AREA_CONFIDENT_THRESHOLD`, the checklist verdict is blended toward a
# neutral midpoint rather than trusted outright, so an unverifiable small
# crop can't outscore a verifiable one without penalizing genuinely
# well-rendered wide shots.
FACE_AREA_CONFIDENT_THRESHOLD = 0.05  # >=5% of frame: checklist trusted at full weight
FACE_AREA_UNCERTAIN_FLOOR = 0.015  # <=1.5% of frame: checklist treated as unverified
UNCERTAIN_SCORE = 0.5  # neutral midpoint, same as the missing-arcface-agreement case


def _small_face_confidence(
    face_box: tuple[int, int, int, int] | None, image_shape: tuple[int, ...]
) -> float:
    """1.0 = the checklist crop was large enough to trust fully, 0.0 = too
    small to mean anything (linear ramp between the two thresholds). A
    missing `face_box` returns 1.0 — that's the "checklist ran on the full
    image" case, handled separately."""
    if face_box is None:
        return 1.0
    x0, y0, x1, y1 = face_box
    height, width = image_shape[0], image_shape[1]
    area_fraction = ((x1 - x0) * (y1 - y0)) / (height * width)
    if area_fraction >= FACE_AREA_CONFIDENT_THRESHOLD:
        return 1.0
    if area_fraction <= FACE_AREA_UNCERTAIN_FLOOR:
        return 0.0
    span = FACE_AREA_CONFIDENT_THRESHOLD - FACE_AREA_UNCERTAIN_FLOOR
    return (area_fraction - FACE_AREA_UNCERTAIN_FLOOR) / span


async def score_identity(
    image_bytes: bytes,
    look_card: LookCard,
    *,
    reference_embedding: np.ndarray | None = None,
) -> tuple[float, tuple[int, int, int, int] | None]:
    """Returns `(score, face_box)`. `score` is in `[0, 1]`, checklist-primary
    with a weak ArcFace agreement term when both a face is detected and a
    `reference_embedding` is given. `face_box` is in the image's own pixel
    coordinates, for the composer's bubble-avoidance — `None` if no face is
    detected, which is not disqualifying on its own (InsightFace/RetinaFace
    can unreliably miss stylized art a VLM reads fine); the checklist still
    runs on the full image.

    When a face is detected but small relative to the frame, the checklist's
    verdict is discounted toward a neutral score — see
    `_small_face_confidence`."""
    image_bgr = decode_jpeg_bgr(image_bytes)
    detected = detect_and_embed(image_bgr)

    checklist_image = (
        encode_crop_jpeg(detected.crop_bgr) if detected else encode_crop_jpeg(image_bgr)
    )
    checklist = await score_checklist(checklist_image, look_card)

    arcface_agreement = None
    if detected is not None and reference_embedding is not None:
        arcface_agreement = (cosine_similarity(detected.embedding, reference_embedding) + 1) / 2

    score = W_CHECKLIST * checklist.score + W_ARCFACE * (
        arcface_agreement if arcface_agreement is not None else 0.5
    )
    face_box = detect_face_box(image_bgr)
    confidence = _small_face_confidence(face_box, image_bgr.shape)
    if confidence < 1.0:
        score = confidence * score + (1 - confidence) * UNCERTAIN_SCORE
    return score, face_box
