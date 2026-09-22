"""Vision judge — a critic signal that can see what the other signals can't.

The critic's other signals are a face checklist, DINOv2 style similarity, SigLIP alignment (near
zero on flat comic art) and edge density. None of them can tell that a deadlift became an
overhead press, that a hand has six fingers, or that the shirt changed colour — the things a
person overrides a strip for. A small vision model can, so one call per panel shows it the
character's reference image and all of the panel's candidates and asks for 1-5 ratings on
five things: scene, face, body, outfit, overall.

Judging the candidates side by side, not one at a time, is deliberate: relative ratings are
far more discriminating than absolute ones. And it is told to ignore the things the person
teaches the app by picking — colour warmth, camera distance, expression strength — so the
judge never encodes a taste of its own on those.

Cost: ~4 images at 512px per panel on Claude Haiku 4.5 is about $0.002 a panel, ~$0.01 a strip.
A failure (network, malformed output) returns no scores; every other signal still works.
"""

from __future__ import annotations

import base64
import logging
from typing import Literal

import cv2
import numpy as np
from pydantic import BaseModel, Field
from storage import Storage

from app.llm.base import ContentPart, Message
from app.llm.routing import resolve
from contracts import Candidate, LookCard, Panel
from ml.identity.faces import decode_jpeg_bgr

logger = logging.getLogger(__name__)

JUDGE_EDGE = 512  # longest edge sent to the judge; smaller = cheaper, 512 still shows hands/outfit
DIMENSIONS = ("scene", "face", "body", "outfit", "overall")
SCORE_KEYS = tuple(f"judge_{d}" for d in DIMENSIONS)
LABELS = "ABCDEF"

Rating = Field(ge=1, le=5)


class CandidateJudgement(BaseModel):
    label: Literal["A", "B", "C", "D", "E", "F"]
    scene: int = Rating
    face: int = Rating
    body: int = Rating
    outfit: int = Rating
    overall: int = Rating


class PanelJudgement(BaseModel):
    candidates: list[CandidateJudgement] = Field(min_length=1, max_length=6)


SYSTEM_PROMPT = (
    "You review candidate images for ONE panel of a personal comic diary, as a strict quality "
    "checker.\n\n"
    "The first image is the reference (model sheet) of the main character. Then come the "
    "candidates for the panel, labelled A, B, C. Rate every candidate from 1 (bad) to 5 "
    "(excellent) on:\n"
    "- scene: does it show what the panel description says — the place, the action, the "
    "objects? 1 = wrong scene or wrong action, 5 = exactly right.\n"
    "- face: is the face natural and clearly the same person as the reference (hair, glasses, "
    "facial hair)? 1 = distorted or a different person.\n"
    "- body: are hands, arms, legs and posture free of anatomical errors (extra or missing "
    "fingers, bent limbs, merged objects)?\n"
    "- outfit: is the character wearing the same clothes as the reference (colour and cut)?\n"
    "- overall: how good is it as a comic frame — clear composition, no stray text or "
    "artefacts?\n\n"
    "Compare the candidates with each other and use the whole 1-5 range; give equal ratings "
    "only when the images really are equal on that point.\n\n"
    "IGNORE these — the person chooses them and they are not quality: colour warmth or palette, "
    "how close or far the camera is, and how strong the facial expression is. Do not reward or "
    "penalise them."
)


def _prepare(storage: Storage, url: str) -> str:
    """The image as a base64 JPEG, longest edge `JUDGE_EDGE`."""
    image = decode_jpeg_bgr(storage.get_object_by_url(url))
    height, width = image.shape[:2]
    scale = JUDGE_EDGE / max(height, width)
    if scale < 1:
        image = cv2.resize(image, (round(width * scale), round(height * scale)))
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("judge: could not encode image")
    return base64.b64encode(np.asarray(buffer).tobytes()).decode()


def build_messages(
    panel: Panel,
    look_card: LookCard | None,
    images_b64: list[str],
    reference_b64: str | None,
) -> list[Message]:
    outfit = look_card.signature_outfit.strip() if look_card else ""
    description = (
        f"Panel description: {panel.action} (place: {panel.place}, time: {panel.time_of_day}, "
        f"caption: {panel.caption_a!r})."
    )
    if outfit and "not enough" not in outfit.lower():
        description += f" The character's usual outfit: {outfit}."
    parts: list[ContentPart] = [{"type": "text", "text": description}]
    if reference_b64:
        parts += [
            {"type": "text", "text": "Reference of the character:"},
            {"type": "image", "b64": reference_b64},
        ]
    for label, encoded in zip(LABELS, images_b64, strict=False):
        parts += [
            {"type": "text", "text": f"Candidate {label}:"},
            {"type": "image", "b64": encoded},
        ]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": parts},
    ]


def normalise(rating: int) -> float:
    """1-5 -> 0-1."""
    return (rating - 1) / 4.0


async def judge_panel(
    panel: Panel,
    candidates: list[Candidate],
    *,
    storage: Storage,
    look_card: LookCard | None,
    reference_url: str | None,
) -> tuple[dict[str, dict[str, float]], float]:
    """`({candidate.id: {judge_scene: .., ...}}, usd)`. Empty scores on any failure — the judge
    is an extra signal, never a reason to lose a strip."""
    if not candidates:
        return {}, 0.0
    try:
        images = [_prepare(storage, c.url) for c in candidates[: len(LABELS)]]
        reference = _prepare(storage, reference_url) if reference_url else None
        provider, model = resolve("critic")
        messages = build_messages(panel, look_card, images, reference)
        result, meta = await provider.structured(
            messages, PanelJudgement, model=model, temperature=0.0
        )
    except Exception:
        logger.exception("judge: panel %s skipped", panel.id)
        return {}, 0.0

    by_label: dict[str, CandidateJudgement] = {j.label: j for j in result.candidates}
    scores: dict[str, dict[str, float]] = {}
    for label, candidate in zip(LABELS, candidates, strict=False):
        judged = by_label.get(label)
        if judged is None:
            continue
        scores[candidate.id] = {f"judge_{d}": normalise(getattr(judged, d)) for d in DIMENSIONS}
    return scores, meta.usd
