"""Master character design — the pivot point of the character-first
identity design: instead of a photo pushed through stylization, the
person approves one designed comic character, and that design becomes the
reference for every future panel (and, later, a character-sheet LoRA).

`fal-ai/flux-2/edit`, a multi-image editing model, given two inputs — the
person's real face crop and a project-wide style reference image
(`services/worker/app/prompts/style_reference.jpg`, defining the same
"flat ink/bold outline" visual language every character in the comic
shares) — redraws the person in that style. This beat every from-scratch
generator tried (Flux LoRA, Leonardo Character Reference, PuLID-Flux) on
both identity fidelity and style fidelity, likely because it edits the
actual photo's pixels rather than generating from a weaker conditioning
signal.

Pushing the prompt hard against a "rounder/fuller face" bias (present, to
some degree, in every mechanism tried) can cause the model to drop color
entirely, reverting to near-monochrome line art — `build_master_prompt`
below is the balance that tested well: explicit but not so heavy-handed on
the negative constraints that it crowds out the style transfer.

`fal-ai/flux-2/edit` charges $0.012/megapixel for input and output
combined, per call. The style reference image is charged on every one of a
batch's calls since it's a real input each time — `_prepare_style_reference`
resizes it once to ~768px on the long edge and caches the uploaded URL for
reuse across a whole batch.
"""

from __future__ import annotations

import asyncio
import io
import os
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from storage import Storage  # noqa: E402

from app.prompts.style_card import StyleCard, load_style_card  # noqa: E402
from contracts import FeatureChecklistResult, LookCard, MasterCandidate  # noqa: E402
from ml.identity.checklist import score_checklist  # noqa: E402
from ml.identity.faces import decode_jpeg_bgr, detect_and_embed, encode_crop_jpeg  # noqa: E402
from ml.identity.fal_jobs import download, submit_and_wait, upload_image  # noqa: E402
from ml.identity.model import IdentityModel, IdentityModelStore  # noqa: E402
from ml.identity.style_score import cosine_to, embed_image  # noqa: E402

MODEL_ID = "fal-ai/flux-2/edit"
STYLE_REFERENCE_PATH = _WORKER_APP_ROOT / "app" / "prompts" / "style_reference.jpg"
STYLE_REFERENCE_MAX_EDGE = 768  # resized before upload — see module docstring's cost note
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5  # the endpoint's own 2.5 default under-corrects the rounder-face bias
# every mechanism tried shows to some degree; 3.5 holds a narrower jaw without dropping color.
OUTPUT_SIZE = {"width": 1024, "height": 1024}
N_CROPS = 3
# 4 seeds x 3 crops = 12 candidates (~$0.50) instead of 24 (~$1.05) for the one-time master
# search. Ranking still returns the top few; override with MASTER_SEEDS_PER_CROP=8 for a
# wider search.
SEEDS_PER_CROP = int(os.environ.get("MASTER_SEEDS_PER_CROP", "4"))
N_CANDIDATES = N_CROPS * SEEDS_PER_CROP
TOP_K = 5  # show the person the top 5 to choose from
DIVERSITY_MAX_COSINE = 0.93  # DINOv2 cosine above which two candidates are "the same picture"
# Ranking weights — a decision, not a tuned constant: the feature checklist
# is the real identity gate, style-match and weak ArcFace agreement are secondary signals.
W_CHECKLIST = 0.5
W_STYLE = 0.3
W_ARCFACE = 0.2

_NONE_VALUES = {"", "none", "none noted", "not enough information"}

CHIP_EDITS: dict[str, tuple[Literal["hair", "glasses", "distinguishing", "face_shape"], str]] = {
    "thicker glasses": ("glasses", "thick, heavy rectangular frames"),
    "thinner glasses": ("glasses", "thin wire frames"),
    "rounder glasses": ("glasses", "round metal frames"),
    "no glasses": ("glasses", "none"),
    "more hair": ("hair", "thick, fuller hair"),
    "shorter hair": ("hair", "short cropped hair"),
    "longer hair": ("hair", "longer hair past the ears"),
    "more facial hair": ("distinguishing", "a full short beard"),
    "clean shaven": ("distinguishing", "clean shaven"),
    "rounder face": ("face_shape", "round"),
    "longer face": ("face_shape", "long, narrow"),
    "narrower face": ("face_shape", "narrow, oval"),
}


class UnknownChipError(ValueError):
    def __init__(self, chip: str) -> None:
        super().__init__(f"apply_chip: unknown chip {chip!r} (see CHIP_EDITS for valid labels)")


class SpendNotConfirmedError(RuntimeError):
    def __init__(self) -> None:
        super().__init__("design_master: pass confirm=True to spend real money generating images")


@dataclass(frozen=True)
class MasterRequest:
    person_id: str
    crops: list[bytes]  # best front crops (jpeg bytes), sharpest first
    look_card: LookCard
    reference_embedding: np.ndarray | None  # IdentityModel.embedding — weak signal only
    seed: int = 0


def _describe_traits(look_card: LookCard) -> str:
    """Same shape as `nodes/prompts.py`'s `_describe_look_card` — stable
    facts stated the same way every time, not left to chance."""
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


# Cycled one-per-seed-offset across a batch (see `generate_candidates`), not left as a single
# fixed expression for every candidate: with an editing model, a fixed prompt plus only a
# varied seed barely changes expression/pose at all (it tends to preserve the source crop's own
# pose), so real variety across a batch has to come from the prompt itself, not seed noise.
POSE_VARIATIONS: list[str] = [
    "a warm smile, facing the viewer, head and shoulders",
    "a thoughtful expression, looking slightly to the side, head and shoulders",
    "a broad laugh, head tilted slightly, head and shoulders",
    "a calm neutral expression, facing the viewer, head and shoulders",
    "an excited grin, facing the viewer, head and shoulders",
    "a relaxed half-smile, three-quarter view, head and shoulders",
    "a curious raised-eyebrow look, facing the viewer, head and shoulders",
    "a confident smile, chin slightly up, head and shoulders",
]


def build_master_prompt(look_card: LookCard, style: StyleCard, *, pose: str | None = None) -> str:
    """See module docstring for the face-shape-vs-color trade-off this
    wording balances. Deliberately does not ask the model to change facial
    structure from the reference image at all (only line/color style), and
    states the "don't round out the face" instruction once, plainly,
    rather than repeating it in a long negative list (repetition pushes
    color out). `pose` defaults to `POSE_VARIATIONS[0]` for a single
    stable call outside a batch (e.g. a one-off preview);
    `generate_candidates` cycles through all of them."""
    traits = _describe_traits(look_card)
    pose = pose or POSE_VARIATIONS[0]
    return (
        f"The first image is a photo of a person: {traits}. The second image is a style "
        f"reference — use it ONLY for its {style.style_phrase}. Do NOT copy the head shape, "
        "face shape, or proportions from the second image. Redraw the person from the first "
        "image in that style, but the face shape, jawline, and cheek contours must exactly "
        "match the first image (the real photo) — do not make the face rounder, fuller, or "
        f"more circular than the actual photo. {pose}, smooth cream background. "
        f"Avoid: {', '.join(style.negative_standard)}, no rounder or fuller face than the photo."
    )


def _resize_max_edge(data: bytes, max_edge: int) -> bytes:
    image = Image.open(io.BytesIO(data)).convert("RGB")
    scale = max_edge / max(image.size)
    if scale < 1:
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


async def _prepare_style_reference(client: httpx.AsyncClient) -> str:
    data = _resize_max_edge(STYLE_REFERENCE_PATH.read_bytes(), STYLE_REFERENCE_MAX_EDGE)
    return await upload_image(data, client=client, file_name="style_reference.jpg")


async def _generate_one(
    crop_jpeg: bytes,
    *,
    crop_index: int,
    seed: int,
    prompt: str,
    style_reference_url: str,
    client: httpx.AsyncClient,
) -> tuple[bytes, str]:
    crop_url = await upload_image(crop_jpeg, client=client, file_name=f"crop-{crop_index}.jpg")
    body = {
        "prompt": prompt,
        "image_urls": [crop_url, style_reference_url],
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "num_images": 1,
        "seed": seed,
        "image_size": OUTPUT_SIZE,
    }
    result = await submit_and_wait(MODEL_ID, body, client=client)
    images = result.get("images", [])
    if not images:
        raise RuntimeError(f"master candidate (crop {crop_index}, seed {seed}): no images returned")
    data = await download(images[0]["url"], client=client)
    return data, prompt


async def generate_candidates(
    request: MasterRequest, style: StyleCard, *, client: httpx.AsyncClient
) -> list[tuple[str, bytes, int, int, str]]:
    """Returns `(id, image_bytes, seed, crop_index, prompt)` tuples — 3
    crops x 8 pose/expression variations (`POSE_VARIATIONS`, one per seed
    offset), so variety across the batch comes from what's actually
    asked for, not from seed noise alone against an editing model that
    tends to preserve the source crop's own pose (see
    `build_master_prompt`'s docstring)."""
    style_reference_url = await _prepare_style_reference(client)

    async def _one(
        crop_index: int, crop_jpeg: bytes, seed_offset: int
    ) -> tuple[str, bytes, int, int, str]:
        # Stride + per-crop shift so fewer seeds per crop (cost cut) still spread
        # across all the pose variations instead of only using the first few.
        stride = max(1, len(POSE_VARIATIONS) // SEEDS_PER_CROP)
        pose = POSE_VARIATIONS[(seed_offset * stride + crop_index) % len(POSE_VARIATIONS)]
        prompt = build_master_prompt(request.look_card, style, pose=pose)
        seed = request.seed + 1000 * crop_index + seed_offset
        data, used_prompt = await _generate_one(
            crop_jpeg,
            crop_index=crop_index,
            seed=seed,
            prompt=prompt,
            style_reference_url=style_reference_url,
            client=client,
        )
        return f"{seed}-{crop_index}", data, seed, crop_index, used_prompt

    tasks = [
        _one(crop_index, crop_jpeg, seed_offset)
        for crop_index, crop_jpeg in enumerate(request.crops[:N_CROPS])
        for seed_offset in range(SEEDS_PER_CROP)
    ]
    return await asyncio.gather(*tasks)


async def rank_candidates(
    raws: list[tuple[str, bytes, int, int, str]],
    request: MasterRequest,
    *,
    style_bank: np.ndarray,
) -> list[MasterCandidate]:
    """Only a confirmed multi-face detection is a hard reject — an
    InsightFace/RetinaFace "no face" result is not treated as one:
    InsightFace, a photo face detector, can fail to detect a face at all
    in clean illustrated art even for a clearly good single-portrait
    render. The feature checklist still runs on the full image in that
    case — a VLM has no trouble seeing a face a photo-trained detector
    misses — only the weak ArcFace signal is unavailable (no aligned
    crop/embedding to compute it from)."""
    ranked = []
    for candidate_id, data, seed, crop_index, prompt in raws:
        image_bgr = decode_jpeg_bgr(data)
        detected = detect_and_embed(image_bgr)

        if detected is not None and detected.face_count > 1:
            # A *confirmed* multi-face detection is trusted even though
            # "no face" isn't — a real second face is a real artifact.
            ranked.append(
                MasterCandidate(
                    id=candidate_id,
                    url="",
                    seed=seed,
                    source_crop_index=crop_index,
                    prompt=prompt,
                    checklist=_empty_checklist(),
                    style_score=0.0,
                    arcface_agreement=None,
                    rank_score=-1.0,
                )
            )
            continue

        checklist_image = (
            encode_crop_jpeg(detected.crop_bgr) if detected else encode_crop_jpeg(image_bgr)
        )
        style = float(np.max(cosine_to(embed_image(image_bgr), style_bank)))
        arcface = None
        if detected is not None and request.reference_embedding is not None:
            from ml.identity.embedding import cosine_similarity

            arcface = (cosine_similarity(detected.embedding, request.reference_embedding) + 1) / 2

        checklist = await score_checklist(checklist_image, request.look_card)
        rank_score = W_CHECKLIST * checklist.score + W_STYLE * style + W_ARCFACE * (arcface or 0.5)

        ranked.append(
            MasterCandidate(
                id=candidate_id,
                url="",
                seed=seed,
                source_crop_index=crop_index,
                prompt=prompt,
                checklist=checklist,
                style_score=style,
                arcface_agreement=arcface,
                rank_score=rank_score,
            )
        )
    return sorted(ranked, key=lambda c: c.rank_score, reverse=True)


def _empty_checklist() -> FeatureChecklistResult:
    """The checklist result stand-in for a candidate that never gets
    scored for real — `rank_candidates`'s confirmed-multi-face-reject
    path short-circuits before the VLM call (pointless to pay for on an
    image already known to have an extra face artifact). A "no face
    detected" result does NOT take this path — see `rank_candidates`'s
    docstring for why."""
    return FeatureChecklistResult(
        checks=[],
        score=0.0,
        mandatory_passed=False,
        artifacts_clean=False,
        artifact_note="no face detected or multiple faces detected in candidate render",
        image_sha256="",
        model="none",
    )


def select_top(
    ranked: list[MasterCandidate], embeddings: dict[str, np.ndarray], *, k: int = TOP_K
) -> list[MasterCandidate]:
    """Greedy diversity pass over gate-passing candidates in rank order:
    accept one only if its DINOv2 cosine to every already-selected
    candidate is < DIVERSITY_MAX_COSINE, otherwise skip it and backfill
    from the skipped ones if fewer than `k` survive — a near-duplicate is
    better than showing the person fewer than `k` options."""
    eligible = [c for c in ranked if c.checklist.mandatory_passed and c.checklist.artifacts_clean]
    selected: list[MasterCandidate] = []
    skipped: list[MasterCandidate] = []
    for candidate in eligible:
        embedding = embeddings.get(candidate.id)
        if embedding is None or not selected:
            selected.append(candidate)
            continue
        max_cosine = max(float(embedding @ embeddings[s.id]) for s in selected)
        if max_cosine < DIVERSITY_MAX_COSINE:
            selected.append(candidate)
        else:
            skipped.append(candidate)
        if len(selected) >= k:
            break
    for candidate in skipped:
        if len(selected) >= k:
            break
        selected.append(candidate)
    return selected[:k]


def apply_chip(look_card: LookCard, chip: str) -> LookCard:
    """Rewrites exactly one look-card field, code-only (no LLM), matching
    the deterministic character_clause approach used elsewhere."""
    if chip not in CHIP_EDITS:
        raise UnknownChipError(chip)
    field, value = CHIP_EDITS[chip]
    current = getattr(look_card, field)
    if field in ("hair",) and current.strip().lower() not in _NONE_VALUES:
        # Merge shape/volume words in without discarding an existing
        # colour word — "more hair" on "black wavy hair" should not lose
        # "black".
        color_words = [w for w in current.split() if w.lower() not in {"hair", "wavy", "straight"}]
        new_value = f"{' '.join(color_words)} {value}".strip()
    else:
        new_value = value
    updated = look_card.model_copy(update={field: new_value, "edits": [*look_card.edits, chip]})
    return updated


async def design_master(
    request: MasterRequest,
    *,
    style: StyleCard | None = None,
    style_bank: np.ndarray | None = None,
    confirm: bool = False,
) -> tuple[list[MasterCandidate], list[MasterCandidate], dict[str, bytes]]:
    """Returns `(top_k, all_ranked, images_by_id)`. Raises
    `SpendNotConfirmedError` unless `confirm=True` — no request is sent
    before that check."""
    if not confirm:
        raise SpendNotConfirmedError
    style = style or load_style_card()
    if style_bank is None:
        from ml.identity.style_score import load_reference_bank

        style_bank = load_reference_bank()

    async with httpx.AsyncClient(timeout=180.0) as client:
        raws = await generate_candidates(request, style, client=client)

    images_by_id = {candidate_id: data for candidate_id, data, *_ in raws}
    ranked = await rank_candidates(raws, request, style_bank=style_bank)
    embeddings = {
        candidate_id: embed_image(decode_jpeg_bgr(data)) for candidate_id, data, *_ in raws
    }
    top = select_top(ranked, embeddings)
    return top, ranked, images_by_id


def approve_master(
    store: IdentityModelStore,
    person_id: str,
    candidate: MasterCandidate,
    image_bytes: bytes,
    *,
    storage: Storage,
    look_card: LookCard,
    style_card_version: str,
    chip_rounds: int,
) -> IdentityModel:
    """Persists the chosen candidate as `people/{id}/master.png` (a
    stable key so a future Character-Reference re-upload or character
    sheet can just read it) and records the six new `IdentityModel`
    fields plus the (possibly chip-edited) look card."""
    key = f"people/{person_id}/master.png"
    url = storage.put_object(key, image_bytes, content_type="image/png")

    existing = store.load(person_id) or IdentityModel(person_id=person_id)
    model = existing.model_copy(
        update={
            "look_card": look_card,
            "master_path": url,
            "master_approved_at": datetime.now(UTC),
            "master_candidate_id": candidate.id,
            "master_seed": candidate.seed,
            "style_card_version": style_card_version,
            "chip_rounds": chip_rounds,
        }
    )
    store.save(model)
    return model
