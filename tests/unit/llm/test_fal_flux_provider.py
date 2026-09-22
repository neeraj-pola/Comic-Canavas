"""FalFluxProvider — mocks the HTTP client methods directly
(`_client.post`/`.get`), same pattern as `test_leonardo_provider.py` and
`test_anthropic_provider.py`. Exercises the real submit/poll/download
code path against canned responses shaped like fal.ai's documented queue
API (see `fal_flux.py`'s module docstring for what's confirmed live).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from storage import LocalFileStorage

from app.images.base import IdentityRef, ImageGenerationError
from app.images.fal_flux import (
    DEV_GUIDANCE_SCALE,
    DEV_MODEL_ID,
    DEV_NUM_INFERENCE_STEPS,
    SCHNELL_MODEL_ID,
    USD_PER_MEGAPIXEL,
    FalFluxProvider,
    MissingFalConfigError,
)
from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from contracts import ImagePrompt


def _response(
    status_code: int, json_data: dict[str, Any] | None = None, content: bytes = b""
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, content=content, request=request)


def _prompt(**overrides: object) -> ImagePrompt:
    defaults: dict[str, object] = {
        "panel_id": 1,
        "generator": "flux",
        "character_clause": "[IDENTITY], content expression",
        "environment_clause": "a chipped mug, morning light, a wooden table, steam",
        "positive": "[IDENTITY], content expression, a chipped mug",
        "negative": "no artist names",
        "seed": 42,
    }
    defaults.update(overrides)
    return ImagePrompt(**defaults)


def _identity(**overrides: object) -> IdentityRef:
    defaults: dict[str, object] = {"trigger_token": "sks person"}
    defaults.update(overrides)
    return IdentityRef(**defaults)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAL_KEY", "test-key-not-real")
    monkeypatch.delenv("IMAGE_FAST", raising=False)


@pytest.fixture
def storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")


@pytest.fixture(autouse=True)
def _cost_sink() -> Iterator[InMemoryCostEventSink]:
    sink = InMemoryCostEventSink()
    previous = get_cost_sink()
    set_cost_sink(sink)
    yield sink
    set_cost_sink(previous)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, storage: LocalFileStorage) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(MissingFalConfigError, match="FAL_KEY"):
        FalFluxProvider(storage)


async def test_generate_happy_path_dev_model(storage: LocalFileStorage) -> None:
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(
                200,
                {
                    "seed": 42,
                    "images": [
                        {"url": "https://cdn.example/1.png", "width": 1024, "height": 768},
                        {"url": "https://cdn.example/2.png", "width": 1024, "height": 768},
                        {"url": "https://cdn.example/3.png", "width": 1024, "height": 768},
                    ],
                },
            ),
            _response(200, content=b"fake1"),
            _response(200, content=b"fake2"),
            _response(200, content=b"fake3"),
        ]
    )

    candidates = await provider.generate(_prompt(), n=3, identity=_identity())

    assert len(candidates) == 3
    assert all(c.panel_id == 1 and c.seed == 42 for c in candidates)
    assert all(c.url.startswith("http://localhost:8000/media/") for c in candidates)

    post_call = provider._client.post.call_args
    assert post_call.args[0] == f"https://queue.fal.run/{DEV_MODEL_ID}"
    body = post_call.kwargs["json"]
    assert body["num_inference_steps"] == DEV_NUM_INFERENCE_STEPS
    assert body["guidance_scale"] == DEV_GUIDANCE_SCALE


async def test_generate_includes_lora_when_identity_has_one(storage: LocalFileStorage) -> None:
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"seed": 1, "images": [{"url": "https://cdn.example/1.png"}]}),
            _response(200, content=b"fake"),
        ]
    )

    await provider.generate(
        _prompt(), n=1, identity=_identity(flux_lora_url="https://r2.example/lora.safetensors")
    )

    body = provider._client.post.call_args.kwargs["json"]
    assert body["loras"] == [{"path": "https://r2.example/lora.safetensors", "scale": 1.0}]


async def test_generate_omits_loras_field_without_a_lora_url(storage: LocalFileStorage) -> None:
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"seed": 1, "images": [{"url": "https://cdn.example/1.png"}]}),
            _response(200, content=b"fake"),
        ]
    )

    await provider.generate(_prompt(), n=1, identity=_identity())

    body = provider._client.post.call_args.kwargs["json"]
    assert "loras" not in body


async def test_generate_uses_schnell_model_and_drops_lora_when_fast(
    monkeypatch: pytest.MonkeyPatch, storage: LocalFileStorage
) -> None:
    monkeypatch.setenv("IMAGE_FAST", "1")
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"seed": 1, "images": [{"url": "https://cdn.example/1.png"}]}),
            _response(200, content=b"fake"),
        ]
    )

    await provider.generate(
        _prompt(), n=1, identity=_identity(flux_lora_url="https://r2.example/lora.safetensors")
    )

    post_call = provider._client.post.call_args
    assert post_call.args[0] == f"https://queue.fal.run/{SCHNELL_MODEL_ID}"
    assert "loras" not in post_call.kwargs["json"]


async def test_poll_uses_returned_urls_for_a_model_id_with_a_slash(
    storage: LocalFileStorage,
) -> None:
    """A reconstructed `{QUEUE_BASE_URL}/{model_id}/requests/{id}/status`
    string is ambiguous when `model_id` contains a `/` — `SCHNELL_MODEL_ID`
    ("fal-ai/flux/schnell") does, and 405s. `_poll_result` must use the
    `status_url`/`response_url` the submit response itself returns."""
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                # Deliberately not of the {QUEUE_BASE_URL}/{model_id}/requests/{id}/...
                # shape, to prove nothing gets reconstructed.
                "status_url": "https://queue.fal.run/some/other/shape/status",
                "response_url": "https://queue.fal.run/some/other/shape/result",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"seed": 1, "images": [{"url": "https://cdn.example/1.png"}]}),
            _response(200, content=b"fake"),
        ]
    )

    await provider.generate(_prompt(), n=1, identity=_identity())

    get_calls = provider._client.get.call_args_list
    assert get_calls[0].args[0] == "https://queue.fal.run/some/other/shape/status"
    assert get_calls[1].args[0] == "https://queue.fal.run/some/other/shape/result"


async def test_generate_records_cost_from_image_megapixels(storage: LocalFileStorage) -> None:
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(
                200,
                {
                    "seed": 1,
                    "images": [{"url": "https://cdn.example/1.png", "width": 1000, "height": 1000}],
                },
            ),
            _response(200, content=b"fake"),
        ]
    )

    sink = get_cost_sink()
    await provider.generate(_prompt(), n=1, identity=_identity())

    assert len(sink.events) == 1  # type: ignore[attr-defined]
    event = sink.events[0]  # type: ignore[attr-defined]
    assert event.provider == "flux"
    assert event.usd == pytest.approx(1.0 * USD_PER_MEGAPIXEL)


async def test_generate_raises_when_no_images_returned(storage: LocalFileStorage) -> None:
    provider = FalFluxProvider(storage)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(
            200,
            {
                "request_id": "req-1",
                "status_url": "https://queue.fal.run/x/requests/req-1/status",
                "response_url": "https://queue.fal.run/x/requests/req-1",
            },
        )
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"seed": 1, "images": []}),
        ]
    )

    with pytest.raises(ImageGenerationError, match="req-1"):
        await provider.generate(_prompt(), n=1, identity=_identity())


def test_estimate_usd_uses_default_megapixels(storage: LocalFileStorage) -> None:
    from app.images.fal_flux import DEFAULT_MEGAPIXELS

    provider = FalFluxProvider(storage)
    assert provider.estimate_usd(n=3) == pytest.approx(DEFAULT_MEGAPIXELS * USD_PER_MEGAPIXEL * 3)
