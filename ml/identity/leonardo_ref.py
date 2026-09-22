"""Leonardo Character Reference upload: uploads a handful of best crops as
Leonardo init images, producing the ids `IdentityModel.leonardo_ref_ids`
stores and `IdentityRef.leonardo_ref_ids` later passes to
`images/leonardo.py`'s `controlnets`.

`POST /init-image` (body `{"extension": "jpg"}`) returns
`{uploadInitImage: {id, fields, key, url}}` — `fields` is a JSON string
(not a nested object) that must be `json.loads`'d. The actual bytes go to
`url` (an S3 presigned POST) as multipart form data: every parsed `fields`
entry plus a `file` field with the image bytes, with no separate
`Authorization` header on this second call since the presigned fields
carry their own auth. `uploadInitImage.id` is the id to keep. Uploading is
free — only generation calls spend credits, not storing reference images.
"""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

BASE_URL = "https://cloud.leonardo.ai/api/rest/v1"


class MissingLeonardoConfigError(ValueError):
    def __init__(self, var: str) -> None:
        super().__init__(f"{var} is not set (see .env.example)")


def _headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


async def upload_init_image(client: httpx.AsyncClient, jpeg_bytes: bytes, *, api_key: str) -> str:
    """Uploads one crop, returns its Leonardo init-image id."""
    create = await client.post(
        f"{BASE_URL}/init-image", headers=_headers(api_key), json={"extension": "jpg"}
    )
    create.raise_for_status()
    upload = create.json()["uploadInitImage"]

    fields: dict[str, Any] = json.loads(upload["fields"])
    put = await client.post(
        upload["url"], data=fields, files={"file": ("crop.jpg", jpeg_bytes, "image/jpeg")}
    )
    put.raise_for_status()

    image_id: str = upload["id"]
    return image_id


async def upload_reference_crops(
    crop_jpegs: list[bytes], *, api_key: str | None = None
) -> list[str]:
    """Uploads each crop in turn, returns the resulting `leonardo_ref_ids`
    in the same order — small batch, a one-time onboarding step, so
    sequential over concurrent is simpler and avoids tripping any
    upload-side rate limit for no real benefit."""
    key = api_key or os.environ.get("LEONARDO_API_KEY")
    if not key:
        raise MissingLeonardoConfigError("LEONARDO_API_KEY")

    async with httpx.AsyncClient(timeout=30.0) as client:
        return [await upload_init_image(client, jpg, api_key=key) for jpg in crop_jpegs]
