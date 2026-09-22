"""Reference-based generation path — the daily-panel generator for the
character-first identity design: renders each panel by referencing the
person's approved `people/{id}/master.png` (locked identity) and the
project's shared `style_reference.jpg` (locked visual style), instead of
training on or re-editing the person's raw photos.

Built on `fal-ai/flux-2/edit`, the same recipe `ml/identity/master.py` uses.
Holding identity and the comic-flat style consistently needs three things on
top of what `nodes/prompts.py` already assembles into the base prompt:

1. **Identity lock.** A sentence naming the first reference image as the
   character's exact identity — without it, the model draws *a* similar
   character each time rather than consistently *the* approved one.
2. **Whole-scene style lock.** The negative-prompt standard alone doesn't
   stop backgrounds rendering semi-realistic; an explicit positive
   instruction to style the ENTIRE image (background included) in the same
   flat-ink/halftone treatment, plus a second style-reference image, closes
   the gap.
3. **Gaze/expression follows the activity.** Without this, the editing model
   defaults to a fixed camera-facing smile regardless of what the character
   is doing.

`ImagePrompt.positive`/`.negative` already carry the character/environment
clauses and the negative standard; `_augment_prompt` adds the three items
above on top, keeping the two-layer prompt system as the single source for
what the scene actually is.

`identity.master_url` and the style reference file are fetched by this
process and re-uploaded to fal.ai's own storage before use, never passed
directly as a `Storage.get_url()` value — fal.ai's remote servers can't
reach a local dev URL, so re-uploading through this worker process (which
can always reach its own storage) works uniformly in dev and prod.
"""

from __future__ import annotations

import asyncio
import io
import os
from pathlib import Path
from typing import Any

import httpx
from PIL import Image
from storage import Storage

from app.images.base import (
    IdentityRef,
    ImageGenerationError,
    download_bytes,
    poll_until,
    variant_tag,
)
from app.llm.base import with_backoff
from app.llm.cost import CostEvent, get_cost_sink
from app.prompts.style_card import StyleCard, load_style_card
from contracts import Candidate, ImagePrompt

QUEUE_BASE_URL = "https://queue.fal.run"
STORAGE_BASE_URL = "https://rest.alpha.fal.ai"
# FLUX.2 Turbo Edit is the default: comparable identity/style fidelity to the
# standard edit model at roughly 33% lower cost per megapixel and 30-40%
# faster. `FAL_EDIT_MODEL=fal-ai/flux-2/edit` restores the standard model.
STANDARD_EDIT_MODEL = "fal-ai/flux-2/edit"
TURBO_EDIT_MODEL = "fal-ai/flux-2/turbo/edit"
# USD per megapixel of input+output, from fal.ai's model pages.
EDIT_MODEL_USD_PER_MEGAPIXEL = {STANDARD_EDIT_MODEL: 0.012, TURBO_EDIT_MODEL: 0.008}
MODEL_ID = os.environ.get("FAL_EDIT_MODEL", TURBO_EDIT_MODEL)
STYLE_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "style_reference.jpg"
STYLE_REFERENCE_MAX_EDGE = 768  # resized before upload — billed on every generation call
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5  # the endpoint's own 2.5 default under-corrects
# `fal-ai/flux-2/edit` bills per megapixel of input AND output, so shrinking
# both the output size and the reference image is a real cost lever; step
# count is not billed. Override with `IMAGE_OUTPUT_EDGE` / `IMAGE_REFERENCE_EDGE`
# (set both to 1024 for sharper output at higher cost).
OUTPUT_EDGE = int(os.environ.get("IMAGE_OUTPUT_EDGE", "768"))
MASTER_REFERENCE_MAX_EDGE = int(os.environ.get("IMAGE_REFERENCE_EDGE", "768"))
OUTPUT_SIZE = {"width": OUTPUT_EDGE, "height": OUTPUT_EDGE}
OUTPUT_MEGAPIXELS = (OUTPUT_SIZE["width"] * OUTPUT_SIZE["height"]) / 1_000_000
# Unknown models fall back to the standard rate (an over-estimate is the safe error).
USD_PER_MEGAPIXEL = EDIT_MODEL_USD_PER_MEGAPIXEL.get(MODEL_ID, 0.012)


def _usd_per_megapixel() -> float:
    return EDIT_MODEL_USD_PER_MEGAPIXEL.get(MODEL_ID, 0.012)


_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError)


class MissingFalConfigError(ImageGenerationError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


class MissingMasterError(ImageGenerationError):
    def __init__(self, trigger_token: str) -> None:
        super().__init__(
            f"IdentityRef.master_url is required for flux_kontext (trigger_token={trigger_token!r}"
            " — approve a master first, task 5.8)"
        )


class FluxKontextProvider:
    """`ImageProvider` for `IMAGE_PROVIDER=flux_kontext` /
    `ImagePrompt.generator == "flux_kontext"`."""

    name = "flux_kontext"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise MissingFalConfigError("FAL_KEY")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=180.0)
        # Both the style reference and any given master_url are stable
        # for the life of this provider instance (one worker process) —
        # uploading either twice would double real fal.ai spend for
        # nothing new, so both are cached after their first upload.
        self._style_reference_url: str | None = None
        self._master_url_cache: dict[str, str] = {}

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"}

    async def _upload(self, data: bytes, *, file_name: str, content_type: str) -> str:
        initiate = await with_backoff(
            lambda: self._client.post(
                f"{STORAGE_BASE_URL}/storage/upload/initiate",
                headers=self._headers(),
                json={"content_type": content_type, "file_name": file_name},
            ),
            retryable=_RETRYABLE,
        )
        initiate.raise_for_status()
        upload = initiate.json()
        put = await self._client.put(
            upload["upload_url"], content=data, headers={"Content-Type": content_type}
        )
        put.raise_for_status()
        file_url: str = upload["file_url"]
        return file_url

    def _prepare_style_reference(self) -> bytes:
        """Resizes the style reference to `STYLE_REFERENCE_MAX_EDGE` before
        upload — it's billed on every generation call, not just its one-time
        upload, so an unnecessarily large reference image doubles real cost
        for no quality gain."""
        image = Image.open(io.BytesIO(STYLE_REFERENCE_PATH.read_bytes())).convert("RGB")
        scale = STYLE_REFERENCE_MAX_EDGE / max(image.size)
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    async def _style_reference_url_cached(self) -> str:
        if self._style_reference_url is None:
            data = self._prepare_style_reference()
            self._style_reference_url = await self._upload(
                data, file_name="style_reference.jpg", content_type="image/jpeg"
            )
        return self._style_reference_url

    async def _fal_reachable_master_url(self, stored_master_url: str) -> str:
        """Fetches the stored master (wherever `Storage.get_url()` says it
        is — local API route or real R2 URL, this process can always
        reach it) and re-uploads the bytes to fal.ai's own storage, so
        fal.ai's servers get a URL *they* can fetch. See module
        docstring's storage note."""
        if stored_master_url in self._master_url_cache:
            return self._master_url_cache[stored_master_url]
        data = self._shrink_master(await download_bytes(self._client, stored_master_url))
        fal_url = await self._upload(data, file_name="master.png", content_type="image/png")
        self._master_url_cache[stored_master_url] = fal_url
        return fal_url

    @staticmethod
    def _shrink_master(data: bytes) -> bytes:
        """Downscales the master to `MASTER_REFERENCE_MAX_EDGE` before the
        upload — it's a billed input on every call. Falls back to the
        original bytes if it can't be decoded (never blocks generation)."""
        try:
            opened = Image.open(io.BytesIO(data))
            opened.load()
        except Exception:
            return data
        scale = MASTER_REFERENCE_MAX_EDGE / max(opened.size)
        if scale >= 1:
            return data
        resized = opened.convert("RGB").resize(
            (int(opened.width * scale), int(opened.height * scale))
        )
        buf = io.BytesIO()
        resized.save(buf, format="PNG", optimize=True)
        return buf.getvalue()

    def _strip_identity_token(self, text: str) -> str:
        """flux_kontext has no LoRA/trigger-word identity signal — identity
        comes entirely from the reference image — so `[IDENTITY]` is removed
        instead of filled in. Strips the trailing ", " too so removing it
        doesn't leave a stray leading comma in the assembled sentence."""
        from app.identity_token import IDENTITY_TOKEN

        return text.replace(f"{IDENTITY_TOKEN}, ", "").replace(IDENTITY_TOKEN, "")

    def _augment_prompt(self, prompt: ImagePrompt, style: StyleCard) -> str:
        """Returns the single, final prompt string sent to the API.
        `fal-ai/flux-2/edit` has no separate `negative_prompt` field —
        negatives fold into the positive text as "Avoid: ...", the same
        convention `build_master_prompt` uses."""
        style_instr = (
            f"Render the ENTIRE image — character, background, and every object — in this "
            f"exact style: {style.style_phrase}. {style.line['style']}, {style.line['weight']}. "
            f"{style.palette['rule']} {style.texture['paper']}. This must look like a hand-drawn "
            f"comic book panel (flat-ink illustration, like a printed comic page), never a "
            f"realistic photo or digital painting."
        )
        lock = (
            "The first reference image is the exact character identity — keep the exact same "
            "face shape, hairstyle, glasses, and any distinguishing features (facial hair "
            "included) as that reference, EVEN under a strained, intense, or exaggerated "
            "expression — a strong expression must never cause a beard, mustache, or other "
            "distinguishing feature to shrink, blur, or disappear. Do not redesign the "
            "character. The character wears the EXACT same outfit as in that reference in every "
            "panel — same garment, colour, collar and fit — and keeps the same body build and "
            "proportions (never heavier, broader, thinner, or baggier than the reference), "
            "unless the scene explicitly requires a costume change. The second reference image "
            "shows the target comic art style for the WHOLE panel including the background — "
            "match it exactly, do not soften or realisticize the background."
        )
        gaze_guard = (
            "The character's gaze and expression must follow what they are doing in the scene, "
            "not a fixed camera-facing smile unrelated to the activity."
        )
        base_positive = self._strip_identity_token(prompt.positive)
        return f"{base_positive} {lock} {style_instr} {gaze_guard} Avoid: {prompt.negative}."

    async def _submit_and_wait(self, body: dict[str, Any]) -> dict[str, Any]:
        response = await with_backoff(
            lambda: self._client.post(
                f"{QUEUE_BASE_URL}/{MODEL_ID}", headers=self._headers(), json=body
            ),
            retryable=_RETRYABLE,
        )
        response.raise_for_status()
        submitted = response.json()
        status_url: str = submitted["status_url"]
        result_url: str = submitted["response_url"]
        request_id: str = submitted["request_id"]

        async def _check() -> dict[str, Any] | None:
            status_response = await self._client.get(status_url, headers=self._headers())
            status_response.raise_for_status()
            if status_response.json().get("status") != "COMPLETED":
                return None
            result_response = await self._client.get(result_url, headers=self._headers())
            result_response.raise_for_status()
            result: dict[str, Any] = result_response.json()
            return result

        return await poll_until(_check, job_id=request_id, timeout_s=300.0)

    def _megapixels(self, image: dict[str, Any]) -> float:
        width = image.get("width")
        height = image.get("height")
        if width and height:
            return float(width) * float(height) / 1_000_000
        return OUTPUT_MEGAPIXELS

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]:
        if not identity.master_url:
            raise MissingMasterError(identity.trigger_token)

        style = load_style_card()
        positive = self._augment_prompt(prompt, style)
        master_url = await self._fal_reachable_master_url(identity.master_url)
        style_url = await self._style_reference_url_cached()

        async def one(i: int) -> Candidate:
            seed = (prompt.seed or 0) + i
            body: dict[str, Any] = {
                "prompt": positive,
                "image_urls": [master_url, style_url],
                "guidance_scale": prompt.guidance or GUIDANCE_SCALE,
                "num_images": 1,
                "seed": seed,
                "image_size": OUTPUT_SIZE,
            }
            # Turbo is distilled with a fixed step count — its endpoint has no
            # `num_inference_steps` input (verified on fal.ai's API page).
            if MODEL_ID == STANDARD_EDIT_MODEL:
                body["num_inference_steps"] = prompt.steps or NUM_INFERENCE_STEPS
            result = await self._submit_and_wait(body)
            images: list[dict[str, Any]] = result.get("images", [])
            if not images:
                raise ImageGenerationError(
                    f"flux_kontext panel {prompt.panel_id}: no images returned"
                )
            image = images[0]
            data = await download_bytes(self._client, image["url"])

            usd = self._megapixels(image) * _usd_per_megapixel()
            get_cost_sink().record(
                CostEvent(provider=self.name, model=MODEL_ID, role="image", units=1, usd=usd)
            )

            key = f"candidates/panel-{prompt.panel_id}/{seed}-{i}-{variant_tag(prompt)}.png"
            url = self.storage.put_object(key, data, content_type="image/png")
            return Candidate(
                id=f"{prompt.panel_id}-{seed}-{i}-{variant_tag(prompt)}",
                panel_id=prompt.panel_id,
                url=url,
                seed=seed,
            )

        # The n images are independent requests: ask for them all at once rather than
        # sequentially, so a panel's images generate concurrently instead of one after another.
        return list(await asyncio.gather(*(one(i) for i in range(n))))

    def estimate_usd(self, *, n: int) -> float:
        # Only the output size is projectable ahead of a call; the two
        # reference-image inputs are cached after their first upload, so
        # their (small, roughly fixed) contribution isn't part of a
        # per-panel marginal estimate the way it is in the real recorded
        # cost event above.
        return OUTPUT_MEGAPIXELS * _usd_per_megapixel() * n
