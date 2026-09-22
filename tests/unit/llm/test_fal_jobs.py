"""ml/identity/fal_jobs.py — node 0's onboarding-time fal.ai calls. Mocks
`httpx.AsyncClient` directly (`client.post`/`.get`/`.put`), same pattern
as `test_leonardo_ref.py`/`test_leonardo_provider.py` rather than the
transport layer.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest

from ml.identity.fal_jobs import MissingFalConfigError, download, submit_and_wait, upload_image


def _response(status_code: int, json_data: dict[str, Any] | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://example.test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, request=request)


async def test_submit_and_wait_polls_until_completed_and_returns_result() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _response(
        200,
        {
            "request_id": "req-1",
            "status_url": "https://queue.fal.run/fal-ai/some-model/requests/req-1/status",
            "response_url": "https://queue.fal.run/fal-ai/some-model/requests/req-1",
        },
    )
    client.get.side_effect = [
        _response(200, {"status": "IN_QUEUE"}),
        _response(200, {"status": "COMPLETED"}),
        _response(200, {"images": [{"url": "https://cdn.example/x.png"}]}),
    ]

    result = await submit_and_wait(
        "fal-ai/some-model", {"prompt": "x"}, client=client, api_key="test-key"
    )

    assert result == {"images": [{"url": "https://cdn.example/x.png"}]}
    client.post.assert_awaited_once()
    assert client.post.call_args.args[0] == "https://queue.fal.run/fal-ai/some-model"


async def test_submit_and_wait_uses_returned_urls_for_a_model_id_with_a_slash() -> None:
    """A reconstructed `{QUEUE_BASE_URL}/{model_id}/requests/{id}/status`
    string is ambiguous when `model_id` itself contains a `/`
    (`fal-ai/flux/dev`) — 405s. `submit_and_wait` must use the URLs the
    submit response itself returns instead."""
    client = AsyncMock(spec=httpx.AsyncClient)
    # Deliberately NOT of the form {QUEUE_BASE_URL}/{model_id}/requests/{id}/... —
    # proves submit_and_wait doesn't reconstruct anything itself.
    client.post.return_value = _response(
        200,
        {
            "request_id": "req-1",
            "status_url": "https://queue.fal.run/some/other/shape/status",
            "response_url": "https://queue.fal.run/some/other/shape/result",
        },
    )
    client.get.side_effect = [
        _response(200, {"status": "COMPLETED"}),
        _response(200, {"images": [{"url": "https://cdn.example/x.png"}]}),
    ]

    result = await submit_and_wait(
        "fal-ai/flux/dev", {"prompt": "x"}, client=client, api_key="test-key"
    )

    assert result == {"images": [{"url": "https://cdn.example/x.png"}]}
    assert client.get.call_args_list[0].args[0] == "https://queue.fal.run/some/other/shape/status"
    assert client.get.call_args_list[1].args[0] == "https://queue.fal.run/some/other/shape/result"


async def test_submit_and_wait_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    client = AsyncMock(spec=httpx.AsyncClient)
    with pytest.raises(MissingFalConfigError, match="FAL_KEY"):
        await submit_and_wait("fal-ai/some-model", {}, client=client)
    client.post.assert_not_awaited()


async def test_upload_image_returns_the_file_url() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.post.return_value = _response(
        200, {"file_url": "https://cdn.example/f.jpg", "upload_url": "https://s3.example/put"}
    )
    client.put.return_value = _response(200)

    url = await upload_image(b"fake-jpeg-bytes", client=client, api_key="test-key")

    assert url == "https://cdn.example/f.jpg"
    put_call = client.put.call_args
    assert put_call.args[0] == "https://s3.example/put"
    assert put_call.kwargs["content"] == b"fake-jpeg-bytes"


async def test_upload_image_raises_without_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    client = AsyncMock(spec=httpx.AsyncClient)
    with pytest.raises(MissingFalConfigError, match="FAL_KEY"):
        await upload_image(b"data", client=client)
    client.post.assert_not_awaited()


async def test_download_returns_response_content() -> None:
    client = AsyncMock(spec=httpx.AsyncClient)
    client.get.return_value = httpx.Response(
        200, content=b"the-bytes", request=httpx.Request("GET", "https://cdn.example/x.png")
    )
    data = await download("https://cdn.example/x.png", client=client)
    assert data == b"the-bytes"
