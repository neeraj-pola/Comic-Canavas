"""fal.ai Flux adapter — Flux Dev + identity LoRA, with a Flux Schnell
fallback for `IMAGE_FAST=1`. Not the default production provider (see
`flux_kontext.py`), kept as a working alternative.

Queue API: `POST https://queue.fal.run/{model_id}` returns `{status,
request_id, status_url, response_url, ...}`; poll `status_url` until
`status == "COMPLETED"`, then fetch `response_url` for the result. Uses the
`status_url`/`response_url` fal's own submit response returns rather than
reconstructing them, since a reconstructed URL is ambiguous whenever
`model_id` itself contains a `/` (`fal-ai/flux/schnell` does).

`fal-ai/flux-lora` (Flux Dev + LoRA) takes `loras: [{path, scale}]`;
`fal-ai/flux/schnell` (the fast path) has no `loras` field — it trades away
identity consistency for speed/cost.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from storage import Storage

from app.identity_token import apply_identity
from app.images.base import (
    IdentityRef,
    ImageGenerationError,
    download_bytes,
    poll_until,
    variant_tag,
)
from app.llm.base import with_backoff
from app.llm.cost import CostEvent, get_cost_sink
from contracts import Candidate, ImagePrompt

QUEUE_BASE_URL = "https://queue.fal.run"
DEV_MODEL_ID = "fal-ai/flux-lora"
SCHNELL_MODEL_ID = "fal-ai/flux/schnell"
DEV_NUM_INFERENCE_STEPS = 28
DEV_GUIDANCE_SCALE = 3.5
USD_PER_MEGAPIXEL = 0.035
# fal-ai/flux-lora's default image_size ("landscape_4_3") is 1024x768.
DEFAULT_MEGAPIXELS = (1024 * 768) / 1_000_000

_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError)


class MissingFalConfigError(ImageGenerationError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


class FalFluxProvider:
    name = "flux"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        api_key = os.environ.get("FAL_KEY")
        if not api_key:
            raise MissingFalConfigError("FAL_KEY")
        self._api_key = api_key
        self.fast = os.environ.get("IMAGE_FAST", "0") == "1"
        # A long-lived client; tests mock `self._client.post`/`.get` directly
        # rather than intercepting at the HTTP transport layer.
        self._client = httpx.AsyncClient(timeout=60.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Key {self._api_key}", "Content-Type": "application/json"}

    def _model_id_and_body(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> tuple[str, dict[str, Any]]:
        positive = apply_identity(prompt.positive, identity.trigger_token)
        if self.fast:
            model_id = SCHNELL_MODEL_ID
            body: dict[str, Any] = {"prompt": positive, "num_images": n}
        else:
            model_id = DEV_MODEL_ID
            body = {
                "prompt": positive,
                "num_images": n,
                "num_inference_steps": prompt.steps or DEV_NUM_INFERENCE_STEPS,
                "guidance_scale": prompt.guidance or DEV_GUIDANCE_SCALE,
            }
            if identity.flux_lora_url:
                body["loras"] = [{"path": identity.flux_lora_url, "scale": 1.0}]
        if prompt.seed is not None:
            body["seed"] = prompt.seed
        return model_id, body

    async def _submit(self, model_id: str, body: dict[str, Any]) -> tuple[str, str, str]:
        """Returns `(request_id, status_url, response_url)` — the latter two
        straight from fal's own submit response, not reconstructed, since a
        reconstructed URL is ambiguous when `model_id` contains a `/`
        (`SCHNELL_MODEL_ID` does)."""
        response = await with_backoff(
            lambda: self._client.post(
                f"{QUEUE_BASE_URL}/{model_id}", headers=self._headers(), json=body
            ),
            retryable=_RETRYABLE,
        )
        response.raise_for_status()
        submitted = response.json()
        return submitted["request_id"], submitted["status_url"], submitted["response_url"]

    async def _poll_result(
        self, request_id: str, status_url: str, result_url: str
    ) -> dict[str, Any]:
        async def _check() -> dict[str, Any] | None:
            response = await self._client.get(status_url, headers=self._headers())
            response.raise_for_status()
            status = response.json()["status"]
            if status == "COMPLETED":
                result_response = await self._client.get(result_url, headers=self._headers())
                result_response.raise_for_status()
                result: dict[str, Any] = result_response.json()
                return result
            return None

        return await poll_until(_check, job_id=request_id)

    def _megapixels(self, images: list[dict[str, Any]]) -> float:
        width = images[0].get("width") if images else None
        height = images[0].get("height") if images else None
        if width and height:
            return float(width) * float(height) / 1_000_000
        return DEFAULT_MEGAPIXELS

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]:
        model_id, body = self._model_id_and_body(prompt, n=n, identity=identity)

        request_id, status_url, result_url = await self._submit(model_id, body)
        result = await self._poll_result(request_id, status_url, result_url)

        images: list[dict[str, Any]] = result.get("images", [])
        if not images:
            raise ImageGenerationError(f"fal.ai request {request_id} produced no images")

        usd = self._megapixels(images) * USD_PER_MEGAPIXEL * len(images)
        get_cost_sink().record(
            CostEvent(provider=self.name, model=model_id, role="image", units=len(images), usd=usd)
        )

        seed = result.get("seed", prompt.seed or 0)
        candidates = []
        for i, image in enumerate(images):
            data = await download_bytes(self._client, image["url"])
            key = f"candidates/panel-{prompt.panel_id}/{seed}-{i}-{variant_tag(prompt)}.png"
            url = self.storage.put_object(key, data, content_type="image/png")
            candidates.append(
                Candidate(id=f"{request_id}-{i}", panel_id=prompt.panel_id, url=url, seed=seed)
            )
        return candidates

    def estimate_usd(self, *, n: int) -> float:
        return DEFAULT_MEGAPIXELS * USD_PER_MEGAPIXEL * n
