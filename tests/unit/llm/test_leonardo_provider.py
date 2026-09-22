"""LeonardoProvider — mocks the HTTP client methods directly
(`_client.post`/`.get`), the same pattern `test_anthropic_provider.py`
uses for the SDK client, rather than intercepting at the transport layer.
Exercises the real request-building/poll-loop/download code path against
canned responses shaped like Leonardo's actual API (the response is
wrapped in `sdGenerationJob`, and cost comes back as a direct dollar
amount, not credits — see `leonardo.py`'s module docstring).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from storage import LocalFileStorage

from app.images.base import IdentityRef, ImageGenerationError
from app.images.leonardo import (
    CHARACTER_REFERENCE_PREPROCESSOR_ID,
    DEFAULT_ESTIMATED_USD_PER_IMAGE,
    DEFAULT_USD_PER_CREDIT,
    LeonardoProvider,
    MissingLeonardoConfigError,
)
from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from contracts import ImagePrompt

CASSETTE = json.loads(
    (Path(__file__).parents[2] / "cassettes" / "leonardo_generation_001.json").read_text()
)


def _response(
    status_code: int, json_data: dict[str, Any] | None = None, content: bytes = b""
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, content=content, request=request)


def _create_response(
    generation_id: str = "gen-1",
    *,
    usd: float | None = None,
    credits: int | None = None,
) -> httpx.Response:
    """Shaped like Leonardo's real `POST /generations` response: wrapped
    in `sdGenerationJob`, cost as a direct dollar amount when known."""
    job: dict[str, Any] = {"generationId": generation_id, "apiCreditCost": credits}
    if usd is not None:
        job["cost"] = {"amount": str(usd), "unit": "DOLLARS"}
    return _response(200, {"sdGenerationJob": job})


def _poll_response(status: str, images: list[dict[str, Any]] | None = None) -> httpx.Response:
    generation: dict[str, Any] = {"status": status}
    if images is not None:
        generation["generated_images"] = images
    return _response(200, {"generations_by_pk": generation})


def _prompt() -> ImagePrompt:
    return ImagePrompt(
        panel_id=1,
        generator="leonardo",
        character_clause="[IDENTITY], content expression",
        environment_clause="a chipped mug, morning light, a wooden table, steam",
        positive="[IDENTITY], content expression, a chipped mug",
        negative="no artist names",
        seed=42,
    )


def _identity(**overrides: object) -> IdentityRef:
    defaults: dict[str, object] = {"trigger_token": "sks person"}
    defaults.update(overrides)
    return IdentityRef(**defaults)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> LocalFileStorage:
    monkeypatch.setenv("LEONARDO_API_KEY", "test-key-not-real")
    monkeypatch.setenv("LEONARDO_MODEL_ID", "test-model-id")
    monkeypatch.delenv("LEONARDO_USD_PER_CREDIT", raising=False)
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")


@pytest.fixture(autouse=True)
def _cost_sink() -> Iterator[InMemoryCostEventSink]:
    sink = InMemoryCostEventSink()
    previous = get_cost_sink()
    set_cost_sink(sink)
    yield sink
    set_cost_sink(previous)


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LEONARDO_API_KEY", raising=False)
    with pytest.raises(MissingLeonardoConfigError, match="LEONARDO_API_KEY"):
        LeonardoProvider(LocalFileStorage(root=tmp_path, base_url="http://localhost:8000"))


def test_missing_model_id_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("LEONARDO_MODEL_ID", raising=False)
    with pytest.raises(MissingLeonardoConfigError, match="LEONARDO_MODEL_ID"):
        LeonardoProvider(LocalFileStorage(root=tmp_path, base_url="http://localhost:8000"))


async def test_generate_happy_path(_env: LocalFileStorage) -> None:
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response(usd=0.015)
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _poll_response(
                "COMPLETE",
                images=[
                    {"id": "img-1", "url": "https://cdn.example/1.png"},
                    {"id": "img-2", "url": "https://cdn.example/2.png"},
                    {"id": "img-3", "url": "https://cdn.example/3.png"},
                ],
            ),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake1"),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake2"),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake3"),
        ]
    )

    candidates = await provider.generate(_prompt(), n=3, identity=_identity())

    assert len(candidates) == 3
    assert [c.id for c in candidates] == ["img-1", "img-2", "img-3"]
    assert all(c.panel_id == 1 and c.seed == 42 for c in candidates)
    assert all(c.url.startswith("http://localhost:8000/media/") for c in candidates)


async def test_generate_replays_the_real_recorded_cassette(_env: LocalFileStorage) -> None:
    """Real POST /generations and GET /generations/{id} responses captured
    against a real account (Phoenix 1.0), replayed offline — `make test`
    never calls the network (pytest-socket), but this exercises the full
    real code path against exactly what Leonardo actually returned."""
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_response(200, CASSETTE["create_response"])
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, CASSETTE["poll_response_complete"]),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake1"),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake2"),
            _response(200, content=b"\x89PNG\r\n\x1a\nfake3"),
        ]
    )

    candidates = await provider.generate(_prompt(), n=3, identity=_identity())

    real_images = CASSETTE["poll_response_complete"]["generations_by_pk"]["generated_images"]
    assert len(candidates) == len(real_images) == 3
    assert {c.id for c in candidates} == {img["id"] for img in real_images}

    sink = get_cost_sink()
    assert sink.events[-1].usd == pytest.approx(0.015)  # type: ignore[attr-defined]


async def test_generate_records_cost_event_from_direct_dollar_amount(
    _env: LocalFileStorage,
) -> None:
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response(usd=0.015)
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _poll_response(
                "COMPLETE", images=[{"id": "img-1", "url": "https://cdn.example/1.png"}]
            ),
            _response(200, content=b"fake"),
        ]
    )

    sink = get_cost_sink()
    await provider.generate(_prompt(), n=1, identity=_identity())

    assert len(sink.events) == 1  # type: ignore[attr-defined]
    event = sink.events[0]  # type: ignore[attr-defined]
    assert event.provider == "leonardo"
    assert event.units == 1
    assert event.usd == pytest.approx(0.015)


async def test_generate_falls_back_to_credits_when_no_dollar_amount(
    _env: LocalFileStorage,
) -> None:
    # Unobserved in practice (see module docstring) but defensive: a plan
    # that returns apiCreditCost with no `cost` object at all.
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response(credits=100)
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _poll_response(
                "COMPLETE", images=[{"id": "img-1", "url": "https://cdn.example/1.png"}]
            ),
            _response(200, content=b"fake"),
        ]
    )

    sink = get_cost_sink()
    await provider.generate(_prompt(), n=1, identity=_identity())

    event = sink.events[0]  # type: ignore[attr-defined]
    assert event.usd == pytest.approx(100 * DEFAULT_USD_PER_CREDIT)


async def test_generate_falls_back_to_flat_estimate_when_no_cost_data(
    _env: LocalFileStorage,
) -> None:
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response()
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _poll_response(
                "COMPLETE", images=[{"id": "img-1", "url": "https://cdn.example/1.png"}]
            ),
            _response(200, content=b"fake"),
        ]
    )

    sink = get_cost_sink()
    await provider.generate(_prompt(), n=1, identity=_identity())

    event = sink.events[0]  # type: ignore[attr-defined]
    assert event.usd == pytest.approx(DEFAULT_ESTIMATED_USD_PER_IMAGE)


async def test_generate_raises_when_status_is_failed(_env: LocalFileStorage) -> None:
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response(credits=10)
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        return_value=_poll_response("FAILED")
    )

    with pytest.raises(ImageGenerationError, match="gen-1"):
        await provider.generate(_prompt(), n=1, identity=_identity())


async def test_generate_polls_across_multiple_pending_checks(
    _env: LocalFileStorage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("asyncio.sleep", AsyncMock())  # skip the real poll interval
    provider = LeonardoProvider(_env)
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        return_value=_create_response(credits=10)
    )
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _poll_response("PENDING"),
            _poll_response("PENDING"),
            _poll_response(
                "COMPLETE", images=[{"id": "img-1", "url": "https://cdn.example/1.png"}]
            ),
            _response(200, content=b"fake"),
        ]
    )

    candidates = await provider.generate(_prompt(), n=1, identity=_identity())
    assert len(candidates) == 1
    assert provider._client.get.await_count == 4


def test_controlnets_include_character_reference_when_ref_ids_present(
    _env: LocalFileStorage,
) -> None:
    provider = LeonardoProvider(_env)
    controlnets = provider._controlnets(_identity(leonardo_ref_ids=["ref-1", "ref-2"]))

    assert controlnets[0]["initImageId"] == "ref-1"
    assert controlnets[0]["preprocessorId"] == CHARACTER_REFERENCE_PREPROCESSOR_ID
    assert controlnets[0]["initImageType"] == "UPLOADED"


def test_controlnets_send_only_the_first_ref_id(_env: LocalFileStorage) -> None:
    """Leonardo 400s ("Multiple Content Transfer or Character Reference
    image guidance inputs not supported.") when more than one Character
    Reference controlnet is sent — only the first uploaded ref id is ever
    actually used."""
    provider = LeonardoProvider(_env)
    controlnets = provider._controlnets(_identity(leonardo_ref_ids=["ref-1", "ref-2", "ref-3"]))
    assert len(controlnets) == 1
    assert controlnets[0]["initImageId"] == "ref-1"


def test_controlnets_empty_when_no_ref_ids(_env: LocalFileStorage) -> None:
    provider = LeonardoProvider(_env)
    assert provider._controlnets(_identity()) == []


def test_estimate_usd_uses_flat_per_image_estimate(_env: LocalFileStorage) -> None:
    provider = LeonardoProvider(_env)
    assert provider.estimate_usd(n=3) == pytest.approx(3 * DEFAULT_ESTIMATED_USD_PER_IMAGE)
