"""Face refinement pass — runs after panel generation on wide/medium framing
(close framing already shows the face large enough to skip by default).
Detects the face region, crops and upscales it, redraws it sharper using
the same reference-locked recipe `flux_kontext.py` uses — locked to the
approved master + the shared style reference — then composites the refined
crop back into the original panel with a feathered edge so the seam is
invisible.

Evaluated against real generated panels and not adopted into the default
pipeline: `flux_kontext.py`'s identity-lock recipe already produces strong
faces at every framing directly, and a second independent stochastic edit
pass on an already-good crop showed real risk of drift with no measured
benefit. Kept as working, tested code but not called by the default
panel-generation path.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any, Literal, cast

import cv2
import httpx
import numpy as np
from PIL import Image
from storage import Storage

from app.images.base import ImageGenerationError, download_bytes, poll_until
from app.llm.base import with_backoff
from app.llm.cost import CostEvent, get_cost_sink
from app.prompts.style_card import StyleCard, load_style_card
from contracts import LookCard
from ml.identity.faces import decode_jpeg_bgr, detect_face_box

QUEUE_BASE_URL = "https://queue.fal.run"
STORAGE_BASE_URL = "https://rest.alpha.fal.ai"
MODEL_ID = "fal-ai/flux-2/edit"
STYLE_REFERENCE_PATH = Path(__file__).resolve().parents[1] / "prompts" / "style_reference.jpg"
STYLE_REFERENCE_MAX_EDGE = 768
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
UPSCALE_FACTOR = 2.5  # applied to the crop's own size, not a fixed resolution
MIN_REFINED_EDGE = 512
MAX_REFINED_EDGE = 1536
USD_PER_MEGAPIXEL = 0.012
FEATHER_PX = 24  # blur radius for the composite edge — hides the seam without
# visibly softening detail well inside the crop

_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError)

DEFAULT_REFINE_FRAMINGS: frozenset[Literal["wide", "medium", "close"]] = frozenset(
    {"wide", "medium"}
)  # close framing already shows the face large enough to skip this


class MissingFalConfigError(ImageGenerationError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


def should_refine(framing: Literal["wide", "medium", "close"], *, force_all: bool = False) -> bool:
    return force_all or framing in DEFAULT_REFINE_FRAMINGS


# `detect_face_box`/`face_bbox_from_kps` live in `ml/identity/faces.py`,
# shared with `critic/identity.py`; re-exported here so existing callers
# importing `detect_face_box` from this module keep working.


def _build_refinement_prompt(look_card: LookCard, style: StyleCard) -> str:
    return (
        "The first reference image is a low-detail crop of a face that needs refining. The "
        "second reference image is the exact character identity — keep the exact same face "
        "shape, hairstyle, glasses, and any distinguishing features as that reference, even "
        "under a strained or exaggerated expression. The third reference image shows the "
        f"target comic art style — match it exactly. Redraw the first image's face sharper "
        f"and higher detail, same pose and expression, in this exact style: "
        f"{style.style_phrase}. {style.line['style']}, {style.line['weight']}. "
        f"{style.palette['rule']} {style.texture['paper']}. Do not change the pose, "
        f"expression, or framing — only increase sharpness and detail. "
        f"Avoid: {', '.join(style.negative_standard)}."
    )


def _feathered_composite(
    panel_bgr: np.ndarray, refined_crop_bgr: np.ndarray, box: tuple[int, int, int, int]
) -> np.ndarray:
    """Composites `refined_crop_bgr` (already resized to the box's exact
    dimensions) into `panel_bgr` at `box`, blending the edge with a
    Gaussian-blurred alpha mask so the seam isn't visible."""
    x1, y1, x2, y2 = box
    mask = np.zeros((y2 - y1, x2 - x1), dtype=np.float32)
    mask[:, :] = 1.0
    # Zero out a thin border so the blur below feathers inward from a
    # true edge, not from a mask that's already 1.0 at the boundary.
    border = max(1, FEATHER_PX // 3)
    mask[:border, :] = 0.0
    mask[-border:, :] = 0.0
    mask[:, :border] = 0.0
    mask[:, -border:] = 0.0
    mask = cast(np.ndarray, cv2.GaussianBlur(mask, (0, 0), sigmaX=FEATHER_PX / 2)).astype(
        np.float32
    )
    mask3 = mask[:, :, None]

    result = panel_bgr.copy()
    region = result[y1:y2, x1:x2].astype(np.float32)
    blended = region * (1 - mask3) + refined_crop_bgr.astype(np.float32) * mask3
    result[y1:y2, x1:x2] = blended.astype(np.uint8)
    return result


class FaceRefiner:
    """Wraps the fal.ai calls needed to refine one face crop — same
    submit/poll/upload shape as `flux_kontext.py`, with its own
    `master_url`/`style_url` caching and instance lifecycle."""

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise MissingFalConfigError("FAL_KEY")
        self._api_key = api_key
        self._client = httpx.AsyncClient(timeout=180.0)
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
        image = Image.open(io.BytesIO(STYLE_REFERENCE_PATH.read_bytes())).convert("RGB")
        scale = STYLE_REFERENCE_MAX_EDGE / max(image.size)
        if scale < 1:
            image = image.resize((int(image.width * scale), int(image.height * scale)))
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()

    async def _style_reference_url_cached(self) -> str:
        if self._style_reference_url is None:
            self._style_reference_url = await self._upload(
                self._prepare_style_reference(),
                file_name="style_reference.jpg",
                content_type="image/jpeg",
            )
        return self._style_reference_url

    async def _fal_reachable_master_url(self, stored_master_url: str) -> str:
        if stored_master_url in self._master_url_cache:
            return self._master_url_cache[stored_master_url]
        data = await download_bytes(self._client, stored_master_url)
        fal_url = await self._upload(data, file_name="master.png", content_type="image/png")
        self._master_url_cache[stored_master_url] = fal_url
        return fal_url

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

        return await poll_until(_check, job_id=request_id, timeout_s=180.0)

    async def refine_crop(
        self, crop_bgr: np.ndarray, *, master_url: str, look_card: LookCard, style: StyleCard
    ) -> np.ndarray:
        """Upscales `crop_bgr` by `UPSCALE_FACTOR` (clamped to a sane
        working range, rounded to a multiple of 8 as fal-ai's models
        expect), redraws it via the reference-locked recipe, and returns
        the refined crop resized back to `crop_bgr`'s original dimensions
        (ready for `_feathered_composite`)."""
        original_size = (crop_bgr.shape[1], crop_bgr.shape[0])
        target_edge = int(max(crop_bgr.shape[:2]) * UPSCALE_FACTOR)
        target_edge = max(MIN_REFINED_EDGE, min(MAX_REFINED_EDGE, target_edge))
        target_edge = (target_edge // 8) * 8

        upscaled = cv2.resize(crop_bgr, (target_edge, target_edge), interpolation=cv2.INTER_CUBIC)
        ok, encoded = cv2.imencode(".jpg", upscaled, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not ok:
            raise ImageGenerationError("refine_crop: failed to encode upscaled crop")

        crop_url = await self._upload(
            encoded.tobytes(), file_name="crop.jpg", content_type="image/jpeg"
        )
        fal_master_url = await self._fal_reachable_master_url(master_url)
        style_url = await self._style_reference_url_cached()

        body = {
            "prompt": _build_refinement_prompt(look_card, style),
            "image_urls": [crop_url, fal_master_url, style_url],
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "num_images": 1,
            "image_size": {"width": target_edge, "height": target_edge},
        }
        result = await self._submit_and_wait(body)
        images: list[dict[str, Any]] = result.get("images", [])
        if not images:
            raise ImageGenerationError("refine_crop: no images returned")

        usd = (target_edge * target_edge * 2) / 1_000_000 * USD_PER_MEGAPIXEL
        get_cost_sink().record(
            CostEvent(
                provider="flux_kontext_refine", model=MODEL_ID, role="image", units=1, usd=usd
            )
        )

        data = await download_bytes(self._client, images[0]["url"])
        refined_bgr = decode_jpeg_bgr(data)
        return cv2.resize(refined_bgr, original_size, interpolation=cv2.INTER_AREA)

    async def refine_panel(
        self,
        panel_bgr: np.ndarray,
        *,
        framing: Literal["wide", "medium", "close"],
        master_url: str,
        look_card: LookCard,
        style: StyleCard | None = None,
        force_all: bool = False,
    ) -> np.ndarray:
        """Top-level entrypoint: returns `panel_bgr` unchanged if
        refinement doesn't apply (close framing, or no face detected) —
        never raises for either of those, only for a real API failure."""
        if not should_refine(framing, force_all=force_all):
            return panel_bgr
        box = detect_face_box(panel_bgr)
        if box is None:
            return panel_bgr

        style = style or load_style_card()
        x1, y1, x2, y2 = box
        crop = panel_bgr[y1:y2, x1:x2]
        refined = await self.refine_crop(
            crop, master_url=master_url, look_card=look_card, style=style
        )
        return _feathered_composite(panel_bgr, refined, box)
