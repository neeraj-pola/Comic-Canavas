"""The provider-agnostic image-generation interface.

Same shape as `llm/base.py`'s `LLMProvider`: one `Protocol` every panel
generator implements, one place any generator SDK/HTTP client is imported
from, `IMAGE_PROVIDER=<name>` picks one. `mock.py` is what tests and the
rest of the pipeline run against without real API keys.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx
from pydantic import BaseModel

from contracts import Candidate, ImagePrompt


class IdentityRef(BaseModel):
    """What a generator needs to render this user (or cast member)
    consistently across panels.

    Exactly one of `leonardo_ref_ids`/`flux_lora_url`/`master_url` is
    populated, matching `ImagePrompt.generator`: Leonardo's Character
    Reference takes uploaded photo ids, Flux's LoRA path takes a
    trained-weights URL, and `master_url` (the character-first path) is the
    person's approved `people/{id}/master.png` used as a locked reference
    image rather than a trigger word — `trigger_token` is unused for it
    (pass `""`); `FluxKontextProvider` derives identity purely from the
    reference image.
    """

    leonardo_ref_ids: list[str] | None = None
    flux_lora_url: str | None = None
    master_url: str | None = None
    trigger_token: str


class ImageGenerationError(Exception):
    """Raised when a provider can't produce any candidates for a prompt."""


class ImageProvider(Protocol):
    name: str

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]: ...

    def estimate_usd(self, *, n: int) -> float:
        """Projected cost of generating `n` candidates for one panel — used
        by the cost guard before any call is made."""
        ...


class PollTimeoutError(ImageGenerationError):
    def __init__(self, *, job_id: str, timeout_s: float) -> None:
        super().__init__(f"job {job_id} did not finish within {timeout_s:.0f}s")


async def poll_until[R](
    check: Callable[[], Awaitable[R | None]],
    *,
    job_id: str,
    interval_s: float = 2.0,
    timeout_s: float = 120.0,
) -> R:
    """Real generators submit a job and poll for its result rather than
    getting it back synchronously — `check` returns the parsed result once
    done, `None` while still in progress."""
    deadline = time.monotonic() + timeout_s
    while True:
        result = await check()
        if result is not None:
            return result
        if time.monotonic() >= deadline:
            raise PollTimeoutError(job_id=job_id, timeout_s=timeout_s)
        await asyncio.sleep(interval_s)


async def download_bytes(client: httpx.AsyncClient, url: str) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content


def variant_tag(prompt: ImagePrompt) -> str:
    """A short, stable tag of a prompt's text, for candidate keys and ids. Candidate files are
    named by seed, but the three options of a panel may share one seed (they differ only in
    their prompt); the tag keeps them from overwriting each other."""
    return hashlib.sha1(prompt.positive.encode()).hexdigest()[:6]
