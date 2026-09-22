"""Character sheet generation — reference-preserving variations generated
from the approved master (`people/{id}/master.png`), not from the
person's raw photos, auto-curated by feature-checklist pass + face-crop
DINOv2-to-master similarity. Feeds the onboarding "meet your character"
screen and the character LoRA's training set.

This targets the KEPT count directly (36, sampled for coverage) rather
than generating a larger batch and culling roughly half — real, avoidable
spend, and risks the eventual LoRA overfitting to near-duplicate poses if
the kept set is padded with redundant near-identical renders.

`VIEWS` states left/right turn direction explicitly (a plain "three-quarter
view" biases toward turning the same way every time), and the sampling
cycles all three evenly so the kept sheet actually shows both sides.

Reuses the identity-lock wording validated in `flux_kontext.py`'s
`_augment_prompt` — an explicit, strongly-worded lock is required to
survive an intense/varied expression without losing a distinguishing
feature (e.g. a mustache).
"""

from __future__ import annotations

import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from storage import Storage  # noqa: E402

from app.prompts.style_card import StyleCard, load_style_card  # noqa: E402
from contracts import LookCard, SheetImage  # noqa: E402
from ml.identity.checklist import score_checklist  # noqa: E402
from ml.identity.faces import decode_jpeg_bgr, detect_and_embed, encode_crop_jpeg  # noqa: E402
from ml.identity.fal_jobs import download, submit_and_wait, upload_image  # noqa: E402
from ml.identity.model import IdentityModel, IdentityModelStore  # noqa: E402
from ml.identity.style_score import (  # noqa: E402
    embed_image,
    identity_similarity,
    master_face_embedding,
)

MODEL_ID = "fal-ai/flux-2/edit"
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5  # same value live-verified for master.py/flux_kontext.py
OUTPUT_SIZE = {"width": 1024, "height": 1024}
N_SHEET = 36  # kept count — see module docstring
DINOV2_KEEP_THRESHOLD = 0.7

_NONE_VALUES = {"", "none", "none noted", "not enough information"}

# Explicitly left/right, not just "three-quarter view" — see module docstring.
VIEWS: list[str] = [
    "facing directly forward, looking at the viewer",
    "turned three-quarters to their own left, so more of the left side of their face is visible",
    "turned three-quarters to their own right, so more of the right side of their face is visible",
]

# 12 distinct expressions, reused (proven wording, same pool `master.py`'s
# POSE_VARIATIONS draws from) across all three views via cycling below —
# 12 x 3 = 36, each view gets even coverage.
EXPRESSIONS: list[str] = [
    "a warm smile",
    "a thoughtful, slightly distant look",
    "a broad open laugh",
    "a calm neutral expression",
    "an excited grin",
    "a relaxed half-smile",
    "a curious raised-eyebrow look",
    "a confident smile, chin slightly up",
    "a gentle closed-mouth smile, soft eyes",
    "a surprised expression, eyebrows raised",
    "a focused, slightly serious expression",
    "a playful smirk, one eyebrow raised",
]

MAX_SHEET = len(VIEWS) * len(EXPRESSIONS)  # every (view, expression) pair is unique up to this


class SpendNotConfirmedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("design_sheet: pass confirm=True to spend real money generating images")


@dataclass(frozen=True)
class SheetRequest:
    person_id: str
    master_jpeg: bytes  # the approved master, re-encoded to real JPEG bytes
    look_card: LookCard
    seed: int = 0


def _describe_traits(look_card: LookCard) -> str:
    """Same shape as `master.py`'s `_describe_traits`/`nodes/prompts.py`'s
    `_describe_look_card` — stable facts stated the same way every time."""
    subject = f"a {look_card.gender_term.strip()}" if look_card.gender_term.strip() else "a person"
    if look_card.age_descriptor.strip():
        gender = look_card.gender_term.strip() or "person"
        subject = f"a {look_card.age_descriptor.strip()} {gender}"
    traits = [subject]
    if look_card.hair.strip():
        traits.append(look_card.hair.strip())
    if look_card.glasses.strip().lower() not in _NONE_VALUES:
        traits.append(look_card.glasses.strip())
    if look_card.distinguishing.strip().lower() not in _NONE_VALUES:
        traits.append(look_card.distinguishing.strip())
    return ", ".join(traits)


def build_sheet_prompt(look_card: LookCard, style: StyleCard, *, view: str, expression: str) -> str:
    """Single-reference prompt (the master itself, already in the right
    style) — unlike `flux_kontext.py`'s daily-panel prompt, no separate
    style-reference image is needed since the master already IS the
    target style; only the identity-lock matters here."""
    traits = _describe_traits(look_card)
    lock = (
        "The reference image is the exact character identity — keep the exact same face shape, "
        "hairstyle, glasses, and any distinguishing features (facial hair included) as that "
        "reference, EVEN under a strained or exaggerated expression. Do not redesign the "
        "character."
    )
    return (
        f"{lock} Redraw this same character: {traits}, {view}, {expression}, head and "
        f"shoulders, plain smooth cream background, in this exact comic style: "
        f"{style.style_phrase}. {style.line['style']}, {style.line['weight']}. "
        f"{style.palette['rule']} {style.texture['paper']}. "
        f"Avoid: {', '.join(style.negative_standard)}."
    )


async def _generate_one(
    master_url: str,
    *,
    view_index: int,
    expression_index: int,
    seed: int,
    prompt: str,
    client: httpx.AsyncClient,
) -> tuple[str, bytes]:
    body = {
        "prompt": prompt,
        "image_urls": [master_url],
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_images": 1,
        "seed": seed,
        "image_size": OUTPUT_SIZE,
    }
    result = await submit_and_wait(MODEL_ID, body, client=client)
    images = result.get("images", [])
    if not images:
        raise RuntimeError(
            f"sheet image (view {view_index}, expr {expression_index}): no images returned"
        )
    data = await download(images[0]["url"], client=client)
    return f"{view_index}-{expression_index}", data


async def generate_sheet_images(
    request: SheetRequest, style: StyleCard, *, n: int = N_SHEET, client: httpx.AsyncClient
) -> list[tuple[str, bytes, str, str]]:
    """Returns `(id, image_bytes, view, expression)` tuples, one per unique
    `(view, expression)` pair via `divmod` — not `i % len(VIEWS)` paired
    with `i % len(EXPRESSIONS)` independently, which collide whenever one
    length divides the other (`len(EXPRESSIONS)` is a multiple of
    `len(VIEWS)` here). `divmod(i, len(EXPRESSIONS))` guarantees a
    genuinely unique pair for every `i` up to `len(VIEWS) * len(EXPRESSIONS)`,
    which `MAX_SHEET` enforces."""
    if n > MAX_SHEET:
        raise ValueError(
            f"generate_sheet_images: n={n} exceeds MAX_SHEET={MAX_SHEET} "
            f"(len(VIEWS) * len(EXPRESSIONS)) — would repeat (view, expression) ids"
        )
    master_url = await upload_image(
        request.master_jpeg, client=client, file_name="master.jpg", content_type="image/jpeg"
    )

    async def _one(i: int) -> tuple[str, bytes, str, str]:
        view_index, expression_index = divmod(i, len(EXPRESSIONS))
        view = VIEWS[view_index]
        expression = EXPRESSIONS[expression_index]
        prompt = build_sheet_prompt(request.look_card, style, view=view, expression=expression)
        seed = request.seed + i
        candidate_id, data = await _generate_one(
            master_url,
            view_index=view_index,
            expression_index=expression_index,
            seed=seed,
            prompt=prompt,
            client=client,
        )
        return candidate_id, data, view, expression

    return await asyncio.gather(*[_one(i) for i in range(n)])


async def score_sheet(
    raws: list[tuple[str, bytes, str, str]], request: SheetRequest, *, master_embedding: np.ndarray
) -> list[SheetImage]:
    """Scores every generated image against the curation rule (checklist
    mandatory pass AND face-crop DINOv2-to-master >= `DINOV2_KEEP_THRESHOLD`)
    — does not drop anything itself, `select_kept` does that from this
    full scored list."""
    scored = []
    for candidate_id, data, view, expression in raws:
        image_bgr = decode_jpeg_bgr(data)
        detected = detect_and_embed(image_bgr)
        checklist_image = (
            encode_crop_jpeg(detected.crop_bgr) if detected else encode_crop_jpeg(image_bgr)
        )
        checklist = await score_checklist(checklist_image, request.look_card)
        dinov2 = identity_similarity(image_bgr, master_embedding)
        dinov2_value = dinov2 if dinov2 is not None else 0.0
        kept = (
            checklist.mandatory_passed
            and checklist.artifacts_clean
            and dinov2_value >= DINOV2_KEEP_THRESHOLD
        )
        scored.append(
            SheetImage(
                id=candidate_id,
                url="",
                view=view,
                expression=expression,
                checklist=checklist,
                dinov2_to_master=dinov2_value,
                kept=kept,
            )
        )
    return scored


def select_kept(scored: list[SheetImage]) -> list[SheetImage]:
    return [s for s in scored if s.kept]


async def design_sheet(
    request: SheetRequest,
    *,
    style: StyleCard | None = None,
    n: int = N_SHEET,
    confirm: bool = False,
) -> tuple[list[SheetImage], list[SheetImage], dict[str, bytes]]:
    """Returns `(kept, all_scored, images_by_id)`. Raises
    `SpendNotConfirmedError` unless `confirm=True` — no request is sent
    before that check."""
    if not confirm:
        raise SpendNotConfirmedError
    style = style or load_style_card()

    master_bgr = decode_jpeg_bgr(request.master_jpeg)
    master_embedding = master_face_embedding(master_bgr)
    if master_embedding is None:
        # Fall back to a whole-image embedding rather than fail outright
        # — should not happen for a real approved master, but a curation
        # run shouldn't hard-crash on it.
        master_embedding = embed_image(master_bgr)

    async with httpx.AsyncClient(timeout=180.0) as client:
        raws = await generate_sheet_images(request, style, n=n, client=client)

    images_by_id = {candidate_id: data for candidate_id, data, *_ in raws}
    scored = await score_sheet(raws, request, master_embedding=master_embedding)
    kept = select_kept(scored)
    return kept, scored, images_by_id


def persist_sheet(
    store: IdentityModelStore,
    person_id: str,
    kept: list[SheetImage],
    images_by_id: dict[str, bytes],
    *,
    storage: Storage,
) -> IdentityModel:
    """Writes every kept image to `people/{id}/sheet/{sheet_image.id}.png`
    and records the resulting keys + timestamp on `IdentityModel`."""
    paths = []
    for sheet_image in kept:
        key = f"people/{person_id}/sheet/{sheet_image.id}.png"
        url = storage.put_object(key, images_by_id[sheet_image.id], content_type="image/png")
        paths.append(url)

    existing = store.load(person_id) or IdentityModel(person_id=person_id)
    model = existing.model_copy(
        update={"sheet_paths": paths, "sheet_generated_at": datetime.now(UTC)}
    )
    store.save(model)
    return model
