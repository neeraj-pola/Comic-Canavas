"""Look card: a vision call over a person's best photo crops, extracting
a short, editable text description (hair, glasses, skin tone, face shape,
signature outfit, distinguishing features).

This is not the identity mechanism itself — that's the image generator's
own reference conditioning. The look card is a stable, plain text
description the person can review and correct themselves, and context the
prompt writer draws on alongside identity conditioning, not instead of it.
See `prompts/lookcard.v1.md` for the full framing.

`LookCard` itself lives in `packages/contracts` (a module boundary between
identity onboarding here and the prompt writer,
`services/worker/app/nodes/prompts.py`), not defined locally.

`gender_term` is on `LookCard` but deliberately not part of what this
vision call asks for or fills in: guessing gender from photos risks
misgendering and isn't more reliable than the person just stating it
themselves, so `_VisionLookCardFields` omits it, and `build_look_card`
takes it as an explicit caller-supplied argument instead.

Bootstraps `services/worker` onto `sys.path` since this calls
`app.llm.routing`/`app.prompts.loader` — `ml/identity/faces.py`/
`embedding.py` don't need this, they have no `app.*` dependency at all.
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from pydantic import BaseModel  # noqa: E402

from app.llm.base import ContentPart, Message  # noqa: E402
from app.llm.routing import resolve  # noqa: E402
from app.prompts.loader import load_prompt  # noqa: E402
from contracts import LookCard  # noqa: E402
from ml.identity.faces import FacePhoto, encode_crop_jpeg  # noqa: E402

MAX_CROPS = 6


class _VisionLookCardFields(BaseModel):
    """The actual structured-output schema for the vision call — the
    same six fields `LookCard` has minus `gender_term`, which this call
    never sees or fills in (see this module's docstring)."""

    hair: str
    glasses: str
    skin_tone: str
    face_shape: str
    signature_outfit: str
    distinguishing: str


class NoUsablePhotosError(ValueError):
    def __init__(self) -> None:
        super().__init__("select_best_crops: no usable photos given")


def select_best_crops(photos: list[FacePhoto], *, n: int = MAX_CROPS) -> list[bytes]:
    """The best crops for the vision call — sharpest first (`blur_score`,
    already computed by face processing), among photos that actually
    passed as usable."""
    usable = [p for p in photos if p.status == "usable" and p.crop_bgr is not None]
    if not usable:
        raise NoUsablePhotosError
    usable.sort(key=lambda p: p.blur_score or 0.0, reverse=True)
    return [encode_crop_jpeg(p.crop_bgr) for p in usable[:n] if p.crop_bgr is not None]


def _image_part(jpeg_bytes: bytes) -> ContentPart:
    return {"type": "image", "b64": base64.b64encode(jpeg_bytes).decode("ascii")}


async def build_look_card(
    crop_jpegs: list[bytes], *, gender_term: str = "", age_descriptor: str = ""
) -> LookCard:
    if not crop_jpegs:
        raise NoUsablePhotosError

    prompt = load_prompt("lookcard")
    provider, model = resolve("lookcard")

    content: list[ContentPart] = [{"type": "text", "text": "Photos of the same person:"}]
    content.extend(_image_part(jpg) for jpg in crop_jpegs)

    messages: list[Message] = [
        {"role": "system", "content": prompt.text},
        {"role": "user", "content": content},
    ]
    fields, _result = await provider.structured(
        messages, _VisionLookCardFields, model=model, temperature=prompt.temperature
    )
    return LookCard(**fields.model_dump(), gender_term=gender_term, age_descriptor=age_descriptor)
