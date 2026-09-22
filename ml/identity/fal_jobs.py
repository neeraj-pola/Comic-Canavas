"""Identity onboarding's single fal.ai entry point (regularization-set and
master-candidate generation) — deliberately separate from
`services/worker/app/images/fal_flux.py`, which is the per-panel
`ImageProvider` bound to `ImagePrompt`/`Candidate` contracts these
one-time onboarding steps don't have. Mirrors `fal_flux.py`'s submit/poll
shape (`POST {QUEUE_BASE_URL}/{model_id}` -> `request_id`; poll
`.../requests/{request_id}/status` until `"COMPLETED"`; fetch
`.../requests/{request_id}` for the result).

Storage upload: `POST {STORAGE_BASE_URL}/storage/upload/initiate` (body
`{content_type, file_name}`) -> `{file_url, upload_url}`; `PUT` the raw
bytes to `upload_url` (no separate auth on this second call — the
presigned URL carries its own). This is the simple upload path, which
413s above roughly 100MB — irrelevant here, every caller uploads a single
face crop or canny edge map, not a multi-hundred-MB file.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import httpx

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.images.base import poll_until  # noqa: E402
from app.llm.base import with_backoff  # noqa: E402

QUEUE_BASE_URL = "https://queue.fal.run"
STORAGE_BASE_URL = "https://rest.alpha.fal.ai"

_RETRYABLE = (httpx.HTTPStatusError, httpx.TransportError)


class MissingFalConfigError(ValueError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


def _api_key(api_key: str | None) -> str:
    key = api_key or os.environ.get("FAL_KEY")
    if not key:
        raise MissingFalConfigError("FAL_KEY")
    return key


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Key {api_key}", "Content-Type": "application/json"}


async def submit_and_wait(
    model_id: str,
    body: dict[str, Any],
    *,
    client: httpx.AsyncClient,
    api_key: str | None = None,
    timeout_s: float = 300.0,
) -> dict[str, Any]:
    """Submits one generation request and blocks until it completes,
    returning the parsed result JSON — same submit/poll shape as
    `images/fal_flux.py`'s `FalFluxProvider`, just without an
    `ImageProvider`/`Candidate` wrapper around it, since onboarding steps
    aren't per-panel generation.

    Uses the `status_url`/`response_url` the submit call itself returns,
    not a reconstructed `{QUEUE_BASE_URL}/{model_id}/requests/{id}/...`
    string, which is ambiguous whenever `model_id` itself contains a `/`
    (e.g. `fal-ai/flux/dev`)."""
    key = _api_key(api_key)
    headers = _headers(key)

    response = await with_backoff(
        lambda: client.post(f"{QUEUE_BASE_URL}/{model_id}", headers=headers, json=body),
        retryable=_RETRYABLE,
    )
    response.raise_for_status()
    submitted = response.json()
    status_url: str = submitted["status_url"]
    result_url: str = submitted["response_url"]
    request_id: str = submitted["request_id"]

    async def _check() -> dict[str, Any] | None:
        status_response = await client.get(status_url, headers=headers)
        status_response.raise_for_status()
        if status_response.json().get("status") != "COMPLETED":
            return None
        result_response = await client.get(result_url, headers=headers)
        result_response.raise_for_status()
        result: dict[str, Any] = result_response.json()
        return result

    return await poll_until(_check, job_id=request_id, timeout_s=timeout_s)


async def upload_image(
    data: bytes,
    *,
    client: httpx.AsyncClient,
    file_name: str = "image.jpg",
    content_type: str = "image/jpeg",
    api_key: str | None = None,
) -> str:
    """Returns a public-but-unguessable `file_url` usable as an
    `image_url`/`control_image_url` in a later request — see this
    module's docstring for the two-call shape and its size limit."""
    key = _api_key(api_key)
    initiate = await client.post(
        f"{STORAGE_BASE_URL}/storage/upload/initiate",
        headers=_headers(key),
        json={"content_type": content_type, "file_name": file_name},
    )
    initiate.raise_for_status()
    upload = initiate.json()
    put = await client.put(
        upload["upload_url"], content=data, headers={"Content-Type": content_type}
    )
    put.raise_for_status()
    file_url: str = upload["file_url"]
    return file_url


async def download(url: str, *, client: httpx.AsyncClient) -> bytes:
    response = await client.get(url)
    response.raise_for_status()
    return response.content
