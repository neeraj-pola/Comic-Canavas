"""Flux LoRA training dataset prep: select the sharpest usable photos,
caption each with a short, non-identity scene description
(background/pose/clothing — never appearance, see
`prompts/trainingcaption.v1.md`'s docstring for why that split matters),
and write an ai-toolkit-format dataset folder: `{i}.jpg` + `{i}.txt` pairs.

Captions follow standard Dreambooth/LoRA practice — the actual class noun
("man"/"woman", not "person") so the trigger token modifies the right one
of Flux's own strong class-conditioned priors, plus a couple of stable,
always-true traits (hair, glasses, distinguishing marks) from the look
card for extra anchoring without reintroducing "two competing constant
signals": `"{trigger_token} a photo of a {gender_term}, {hair}, {glasses},
{distinguishing}, {description}"` (glasses/distinguishing omitted when
"none"/"none noted").
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.llm.base import ContentPart, LLMProvider, Message  # noqa: E402
from app.llm.routing import resolve  # noqa: E402
from app.prompts.loader import Prompt, load_prompt  # noqa: E402
from contracts import LookCard, SheetImage  # noqa: E402
from ml.identity.faces import FacePhoto, encode_crop_jpeg, largest_face_kps  # noqa: E402

_NONE_VALUES = {"", "none", "none noted", "not enough information"}

# Some vision models refuse an innocuous background-only question about a
# tightly-cropped face photo even though nothing about identity was asked.
# Rather than failing the whole dataset build over a false-positive safety
# refusal, detect it and retry once with a specific fallback provider.
_REFUSAL_MARKERS = ("i'm sorry", "i cannot", "i can't", "can't help", "unable to help")
_FALLBACK_MODEL = "claude-sonnet-4-6"

# A lower, "don't bother, there's not enough signal" floor for cases (like
# a smaller real onboarding batch) where the ideal 15-25 range isn't
# available — callers should still surface the gap, not silently substitute.
MIN_TRAINING_PHOTOS = 8
MAX_TRAINING_PHOTOS = 25


class TooFewTrainingPhotosError(ValueError):
    def __init__(self, count: int) -> None:
        super().__init__(
            f"only {count} usable photos available, need at least "
            f"{MIN_TRAINING_PHOTOS} to train a LoRA"
        )


def select_training_photos(
    photos: list[FacePhoto], *, n: int = MAX_TRAINING_PHOTOS
) -> list[FacePhoto]:
    usable = [p for p in photos if p.status == "usable"]
    usable.sort(key=lambda p: p.blur_score or 0.0, reverse=True)
    return usable[:n]


def _looks_like_a_refusal(text: str) -> bool:
    lowered = text.strip().lower()
    return len(lowered) < 15 or any(marker in lowered for marker in _REFUSAL_MARKERS)


async def _caption_once(
    provider: LLMProvider, model: str, prompt: Prompt, jpeg_bytes: bytes
) -> str:
    content: list[ContentPart] = [
        {"type": "text", "text": prompt.text},
        {"type": "image", "b64": base64.b64encode(jpeg_bytes).decode("ascii")},
    ]
    messages: list[Message] = [{"role": "user", "content": content}]
    result = await provider.complete(messages, model=model, temperature=prompt.temperature)
    return result.text.strip()


async def caption_training_photo(jpeg_bytes: bytes) -> str:
    """One short, non-identity scene description — see this module's
    docstring and `prompts/trainingcaption.v1.md` for why identity is
    deliberately excluded here. Falls back to a fallback provider once if
    the primary provider's response looks like a refusal."""
    prompt = load_prompt("trainingcaption")
    provider, model = resolve("lookcard")
    caption = await _caption_once(provider, model, prompt, jpeg_bytes)

    if _looks_like_a_refusal(caption):
        from app.llm.anthropic_provider import AnthropicProvider

        caption = await _caption_once(AnthropicProvider(), _FALLBACK_MODEL, prompt, jpeg_bytes)

    return caption


def _caption_prefix(trigger_token: str, gender_term: str, look_card: LookCard | None) -> str:
    """The fixed, identity-anchoring part of every training caption — the
    same for every photo (like the trigger token itself), which is exactly
    why it's safe to include here and not in the per-photo scene
    description.

    Mirrors `nodes/prompts.py`'s `_describe_look_card` (age, hair, glasses,
    distinguishing — `face_shape` is skipped here since ai-toolkit's
    captions read better dense with concrete traits than with an add-on
    "X face" phrase)."""
    class_noun = gender_term.strip() or "person"
    if look_card is not None and look_card.age_descriptor.strip():
        class_noun = f"{look_card.age_descriptor.strip()} {class_noun}"
    traits = [class_noun]
    if look_card is not None:
        if look_card.hair.strip():
            traits.append(look_card.hair.strip())
        if look_card.glasses.strip().lower() not in _NONE_VALUES:
            traits.append(look_card.glasses.strip())
        if look_card.distinguishing.strip().lower() not in _NONE_VALUES:
            traits.append(look_card.distinguishing.strip())
    return f"{trigger_token} a photo of a {', '.join(traits)}"


async def build_training_dataset(
    photos: list[FacePhoto],
    output_dir: Path,
    *,
    trigger_token: str,
    gender_term: str = "",
    look_card: LookCard | None = None,
    n: int = MAX_TRAINING_PHOTOS,
) -> int:
    """Writes `{i}.jpg` + `{i}.txt` pairs into `output_dir` (ai-toolkit's
    expected dataset layout). Returns the number of pairs written.

    `gender_term`/`look_card` anchor every caption's fixed prefix (see
    this module's docstring) — both optional and default to a generic
    "person" with no extra traits, for callers that don't have a look card
    yet."""
    selected = select_training_photos(photos, n=n)
    if len(selected) < MIN_TRAINING_PHOTOS:
        raise TooFewTrainingPhotosError(len(selected))

    prefix = _caption_prefix(trigger_token, gender_term, look_card)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, photo in enumerate(selected):
        assert photo.crop_bgr is not None  # "usable" always carries a crop
        jpeg = encode_crop_jpeg(photo.crop_bgr)
        (output_dir / f"{i}.jpg").write_bytes(jpeg)

        description = await caption_training_photo(jpeg)
        caption = f"{prefix}, {description}"
        (output_dir / f"{i}.txt").write_text(caption)

    return len(selected)


# --- Character-sheet dataset ------------------------------------------
#
# Deliberately the opposite captioning convention from the photo LoRA
# above: that one needed identity words in every caption because
# appearance wasn't otherwise guaranteed consistent across a person's raw
# photos. Every sheet image already IS the one approved, curated
# character — identity is invariant across the whole dataset, so per
# standard Dreambooth/LoRA practice it should not be restated per-caption
# (that would fight the trigger token for the "constant signal" role, the
# same problem the other way round). Captions here describe only what
# actually varies (view/expression) plus fixed filler for axes the sheet
# didn't vary (outfit/background — both constant across the real sheet).

CHARACTER_TRIGGER_TOKEN = "nrj_cc"  # distinct from IdentityModel.trigger_token (the separate
# photo-LoRA experiment) — a new token for a new, separate model.
DEFAULT_OUTFIT = "a black t-shirt"  # the sheet didn't vary outfit; this is what was actually
# generated, stated honestly as a constant rather than invented per-image variety.
DEFAULT_BACKGROUND = "a plain cream background"  # same reasoning — the sheet's actual background.

FACE_MASK_VALUE = 255  # 1.0 — full loss weight on the face region
BACKGROUND_MASK_VALUE = 102  # ~0.4 — a 255/102 ≈ 2.5x face:background ratio, expressed as a
# pixel-value ratio in the mask image itself (ai-toolkit's `mask_path`: "white has higher loss
# than black", 0-1 range).


def _face_region_mask(shape: tuple[int, int], kps: np.ndarray | None) -> np.ndarray:
    """An approximate elliptical face-region mask for weighted loss — not
    precise segmentation, a soft loss-weighting hint. `kps` (5 landmarks:
    two eyes, nose, two mouth corners) spans only the central face, so the
    ellipse is padded well beyond that raw spread to cover
    forehead/cheeks/chin/ears. Falls back to an all-background (no
    weighting) mask if no face was detected — the same InsightFace/
    RetinaFace limitation on stylized art documented elsewhere in this
    codebase, not a reason to fail the whole dataset build over one image."""
    height, width = shape[:2]
    mask = np.full((height, width), BACKGROUND_MASK_VALUE, dtype=np.uint8)
    if kps is None:
        return mask

    xs, ys = kps[:, 0], kps[:, 1]
    center_x, center_y = float(xs.mean()), float(ys.mean())
    spread_x = float(xs.max() - xs.min())
    spread_y = float(ys.max() - ys.min())
    radius_x = max(spread_x * 1.8, spread_y * 1.4, 1.0)
    radius_y = max(spread_y * 2.2, spread_x * 1.6, 1.0)
    # Shift the ellipse center up slightly — kps's vertical centroid sits
    # near the nose/mouth, but a face region needs more room above (the
    # forehead) than below (the chin, already close to the mouth).
    cv2.ellipse(
        mask,
        (int(center_x), int(center_y - radius_y * 0.15)),
        (int(radius_x), int(radius_y)),
        0,
        0,
        360,
        FACE_MASK_VALUE,
        -1,
    )
    return mask


def build_character_dataset(
    sheet_images: list[SheetImage],
    images_by_id: dict[str, bytes],
    output_dir: Path,
    *,
    trigger_token: str = CHARACTER_TRIGGER_TOKEN,
    outfit: str = DEFAULT_OUTFIT,
    background: str = DEFAULT_BACKGROUND,
) -> int:
    """Writes `{i}.jpg` + `{i}.txt` (ai-toolkit's dataset layout, same as
    v1) plus `masks/{i}.png` (ai-toolkit's `mask_path` layout — same
    filenames, sibling folder) for every kept sheet image. Returns the
    number of triples written."""
    from ml.identity.faces import decode_jpeg_bgr

    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    for i, sheet_image in enumerate(sheet_images):
        data = images_by_id[sheet_image.id]
        (output_dir / f"{i}.jpg").write_bytes(data)

        caption = (
            f"{trigger_token}, comic character, {sheet_image.view}, "
            f"{sheet_image.expression}, {outfit}, {background}"
        )
        (output_dir / f"{i}.txt").write_text(caption)

        image_bgr = decode_jpeg_bgr(data)
        kps = largest_face_kps(image_bgr)
        mask = _face_region_mask(image_bgr.shape, kps)
        cv2.imwrite(str(masks_dir / f"{i}.png"), mask)

    return len(sheet_images)
