"""Leonardo.ai adapter — Phoenix/Lucid Origin models with Character
Reference for identity-consistent panels. Not the default production
provider (see `flux_kontext.py`), kept as a working alternative.

`LEONARDO_MODEL_ID` must be the model's UUID from `GET /platformModels`
(e.g. Phoenix 1.0 is `de7d3faf-762f-48e0-b3b7-9d0ac3a3fcf3`) — a
human-readable name like `"phoenix-v1.0"` 500s instead of erroring clearly.

`POST /generations`'s response is wrapped in `sdGenerationJob`. Cost comes
back as a direct dollar amount (`cost.amount`/`cost.unit == "DOLLARS"`), with
`apiCreditCost`-based estimation only as a fallback for plans that return
credits instead. Character Reference (`controlnets`) needs Phoenix's own
preprocessor id (397, not SDXL's 133) and accepts at most one reference
image per generation.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from storage import Storage

from app.identity_token import apply_identity
from app.images.base import IdentityRef, ImageGenerationError, download_bytes, poll_until
from app.llm.base import with_backoff
from app.llm.cost import CostEvent, get_cost_sink
from contracts import Candidate, ImagePrompt

BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"
# Phoenix's own Character Reference preprocessor id (SDXL's is 133 and 400s
# against this model).
CHARACTER_REFERENCE_PREPROCESSOR_ID = 397
DEFAULT_STRENGTH_TYPE = "Mid"

# Fallback path only — real responses give a direct dollar `cost.amount`;
# these are for a plan that returns `apiCreditCost` (credits) instead.
DEFAULT_USD_PER_CREDIT = 0.01
DEFAULT_ESTIMATED_CREDITS_PER_IMAGE = 20
DEFAULT_ESTIMATED_USD_PER_IMAGE = 0.01

_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError)


class MissingLeonardoConfigError(ImageGenerationError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


class LeonardoProvider:
    name = "leonardo"

    def __init__(self, storage: Storage) -> None:
        self.storage = storage
        api_key = os.environ.get("LEONARDO_API_KEY")
        if not api_key:
            raise MissingLeonardoConfigError("LEONARDO_API_KEY")
        model_id = os.environ.get("LEONARDO_MODEL_ID")
        if not model_id:
            raise MissingLeonardoConfigError("LEONARDO_MODEL_ID")
        self._api_key = api_key
        self.model_id = model_id
        # A long-lived client; tests mock `self._client.post`/`.get` directly
        # rather than intercepting at the HTTP transport layer.
        self._client = httpx.AsyncClient(timeout=60.0)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}

    def _controlnets(self, identity: IdentityRef) -> list[dict[str, Any]]:
        # Only the first ref id — Leonardo rejects more than one Character
        # Reference controlnet per generation. `leonardo_ref_ids` keeps all
        # uploaded crops on the identity record regardless (e.g. for a
        # future ref-selection UI); this is the one call site that turns
        # them into an actual request.
        if not identity.leonardo_ref_ids:
            return []
        return [
            {
                "initImageId": identity.leonardo_ref_ids[0],
                "initImageType": "UPLOADED",
                "preprocessorId": CHARACTER_REFERENCE_PREPROCESSOR_ID,
                "strengthType": DEFAULT_STRENGTH_TYPE,
            }
        ]

    async def _create_generation(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> tuple[str, float | None]:
        positive = apply_identity(prompt.positive, identity.trigger_token)
        body: dict[str, Any] = {
            "modelId": self.model_id,
            "prompt": positive,
            "negative_prompt": prompt.negative,
            "num_images": n,
            "controlnets": self._controlnets(identity),
        }
        if prompt.seed is not None:
            body["seed"] = prompt.seed
        if prompt.guidance is not None:
            body["guidance_scale"] = prompt.guidance

        response = await with_backoff(
            lambda: self._client.post(
                f"{BASE_URL}/generations", headers=self._headers(), json=body
            ),
            retryable=_RETRYABLE,
        )
        response.raise_for_status()
        job = response.json()["sdGenerationJob"]
        return job["generationId"], self._usd_from_job(job)

    def _usd_from_job(self, job: dict[str, Any]) -> float | None:
        """Prefers the direct dollar amount Leonardo returns
        (`cost.amount`/`cost.unit == "DOLLARS"`); `apiCreditCost` (credits,
        needing `LEONARDO_USD_PER_CREDIT` to convert) is a fallback for
        plans that return credits instead."""
        cost = job.get("cost")
        if cost and cost.get("unit") == "DOLLARS":
            try:
                return float(cost["amount"])
            except (TypeError, ValueError):
                pass
        credits = job.get("apiCreditCost")
        if credits is not None:
            usd_per_credit = float(
                os.environ.get("LEONARDO_USD_PER_CREDIT", DEFAULT_USD_PER_CREDIT)
            )
            return float(credits) * usd_per_credit
        return None

    async def _poll_result(self, generation_id: str) -> list[dict[str, Any]]:
        async def _check() -> list[dict[str, Any]] | None:
            response = await self._client.get(
                f"{BASE_URL}/generations/{generation_id}", headers=self._headers()
            )
            response.raise_for_status()
            generation = response.json()["generations_by_pk"]
            status = generation["status"]
            if status == "FAILED":
                raise ImageGenerationError(f"Leonardo generation {generation_id} failed")
            if status == "COMPLETE":
                images: list[dict[str, Any]] = generation["generated_images"]
                return images
            return None

        return await poll_until(_check, job_id=generation_id)

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]:
        generation_id, usd_from_job = await self._create_generation(prompt, n=n, identity=identity)
        images = await self._poll_result(generation_id)
        if not images:
            raise ImageGenerationError(f"Leonardo generation {generation_id} had no images")

        usd = usd_from_job if usd_from_job is not None else self.estimate_usd(n=len(images))
        get_cost_sink().record(
            CostEvent(
                provider=self.name,
                model=self.model_id,
                role="image",
                units=len(images),
                usd=usd,
            )
        )

        candidates = []
        for image in images:
            data = await download_bytes(self._client, image["url"])
            key = f"candidates/panel-{prompt.panel_id}/{image['id']}.png"
            url = self.storage.put_object(key, data, content_type="image/png")
            candidates.append(
                Candidate(
                    id=str(image["id"]),
                    panel_id=prompt.panel_id,
                    url=url,
                    seed=prompt.seed or 0,
                )
            )
        return candidates

    def estimate_usd(self, *, n: int) -> float:
        return DEFAULT_ESTIMATED_USD_PER_IMAGE * n
