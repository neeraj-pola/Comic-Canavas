"""ml/identity/leonardo_ref.py — uploading crops as Leonardo init images.
Mocks the HTTP client directly (`client.post`), same pattern as
`test_leonardo_provider.py`, rather than the transport layer. Uploading
itself is free, per the module's own docstring.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from ml.identity.leonardo_ref import (
    MissingLeonardoConfigError,
    upload_init_image,
    upload_reference_crops,
)


def _response(status_code: int, json_data: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, request=request)


def _create_response(image_id: str) -> httpx.Response:
    return _response(
        200,
        {
            "uploadInitImage": {
                "id": image_id,
                "fields": f'{{"key": "users/x/initImages/{image_id}.jpg", "bucket": "b"}}',
                "key": f"users/x/initImages/{image_id}.jpg",
                "url": "https://s3.example/bucket",
            }
        },
    )


async def test_upload_init_image_returns_the_id() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = [_create_response("img-1"), _response(204)]

    image_id = await upload_init_image(client, b"fake-jpeg-bytes", api_key="test-key")

    assert image_id == "img-1"
    assert client.post.call_count == 2


async def test_upload_init_image_posts_parsed_fields_and_file_to_presigned_url() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.side_effect = [_create_response("img-2"), _response(204)]

    await upload_init_image(client, b"fake-jpeg-bytes", api_key="test-key")

    second_call = client.post.call_args_list[1]
    assert second_call.args[0] == "https://s3.example/bucket"
    assert second_call.kwargs["data"] == {"key": "users/x/initImages/img-2.jpg", "bucket": "b"}
    assert second_call.kwargs["files"]["file"][1] == b"fake-jpeg-bytes"


async def test_upload_reference_crops_uploads_each_crop_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LEONARDO_API_KEY", "test-key-not-real")
    responses = [
        _create_response("img-1"),
        _response(204),
        _create_response("img-2"),
        _response(204),
    ]

    class _FakeClient:
        def __init__(self, **kwargs: object) -> None:
            self._responses = iter(responses)

        async def post(self, *args: object, **kwargs: object) -> httpx.Response:
            return next(self._responses)

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc_info: object) -> None:
            return None

    monkeypatch.setattr(httpx, "AsyncClient", _FakeClient)

    ids = await upload_reference_crops([b"crop-1", b"crop-2"])
    assert ids == ["img-1", "img-2"]


async def test_upload_reference_crops_raises_without_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LEONARDO_API_KEY", raising=False)
    with pytest.raises(MissingLeonardoConfigError, match="LEONARDO_API_KEY"):
        await upload_reference_crops([b"crop-1"])
