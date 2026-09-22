"""FluxKontextProvider — mocks the HTTP client methods directly, same
pattern as `test_fal_flux_provider.py`. Exercises the real
fetch-master -> upload-to-fal -> submit -> poll -> download code path
against canned responses shaped like fal.ai's documented queue/storage
APIs (see `flux_kontext.py`'s module docstring for what's live-verified).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import pytest
from storage import LocalFileStorage

from app.images import flux_kontext
from app.images.base import IdentityRef, ImageGenerationError
from app.images.flux_kontext import (
    GUIDANCE_SCALE,
    NUM_INFERENCE_STEPS,
    OUTPUT_MEGAPIXELS,
    USD_PER_MEGAPIXEL,
    FluxKontextProvider,
    MissingFalConfigError,
    MissingMasterError,
)
from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from app.prompts.style_card import load_style_card
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
        "generator": "flux_kontext",
        "character_clause": "[IDENTITY], a man, short black hair, content expression",
        "environment_clause": "a chipped mug, morning light, a wooden table, steam",
        "positive": (
            "[IDENTITY], a man, short black hair, content expression. a chipped mug. wide shot."
        ),
        "negative": "no photorealism, no gradients",
        "seed": 42,
    }
    defaults.update(overrides)
    return ImagePrompt(**defaults)


def _identity(**overrides: object) -> IdentityRef:
    defaults: dict[str, object] = {
        "trigger_token": "",
        "master_url": "http://localhost:8000/media/people/me/master.png",
    }
    defaults.update(overrides)
    return IdentityRef(**defaults)


@pytest.fixture(autouse=True)
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FAL_KEY", "test-key-not-real")


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
        FluxKontextProvider(storage)


async def test_generate_raises_without_a_master_url(storage: LocalFileStorage) -> None:
    provider = FluxKontextProvider(storage)
    with pytest.raises(MissingMasterError):
        await provider.generate(_prompt(), n=1, identity=_identity(master_url=None))


def test_augment_prompt_strips_identity_token_and_adds_locks() -> None:
    provider = FluxKontextProvider.__new__(FluxKontextProvider)  # no client/API key needed
    style = load_style_card()
    positive = provider._augment_prompt(_prompt(), style)

    assert "[IDENTITY]" not in positive
    assert positive.startswith("a man, short black hair")  # no leading comma artifact
    assert "first reference image is the exact character identity" in positive
    assert "second reference image shows the target comic art style" in positive
    assert "gaze and expression must follow" in positive
    # fal-ai/flux-2/edit has no negative_prompt field (master.py, task
    # 5.8) — negatives must fold into the positive text, not get dropped.
    assert "Avoid: no photorealism, no gradients." in positive


def test_strip_identity_token_handles_token_with_no_trailing_comma() -> None:
    provider = FluxKontextProvider.__new__(FluxKontextProvider)
    assert provider._strip_identity_token("[IDENTITY]") == ""
    # No ", " to match, but the bare-token fallback still removes it —
    # the token must never survive into the final prompt either way.
    assert provider._strip_identity_token("a man [IDENTITY]") == "a man "
    assert provider._strip_identity_token("[IDENTITY], a man") == "a man"


@pytest.mark.parametrize("model", [flux_kontext.TURBO_EDIT_MODEL, flux_kontext.STANDARD_EDIT_MODEL])
async def test_generate_happy_path(
    storage: LocalFileStorage, monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setattr(flux_kontext, "MODEL_ID", model)
    provider = FluxKontextProvider(storage)
    # Call order for n=1: GET master bytes -> POST+PUT upload master ->
    # POST+PUT upload style reference -> POST submit -> GET status ->
    # GET result -> GET final image bytes.
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, content=b"fake-master-bytes"),
            _response(200, {"status": "COMPLETED"}),
            _response(
                200,
                {"images": [{"url": "https://cdn.example/1.png", "width": 1024, "height": 1024}]},
            ),
            _response(200, content=b"fake-final-image"),
        ]
    )
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(
                200, {"file_url": "https://fal.media/master.png", "upload_url": "https://up/m"}
            ),
            _response(
                200, {"file_url": "https://fal.media/style.jpg", "upload_url": "https://up/s"}
            ),
            _response(
                200,
                {
                    "request_id": "req-1",
                    "status_url": "https://queue.fal.run/x/requests/req-1/status",
                    "response_url": "https://queue.fal.run/x/requests/req-1",
                },
            ),
        ]
    )
    provider._client.put = AsyncMock(return_value=_response(200))  # type: ignore[method-assign]

    candidates = await provider.generate(_prompt(), n=1, identity=_identity())

    assert len(candidates) == 1
    assert candidates[0].panel_id == 1
    assert candidates[0].seed == 42
    assert candidates[0].url.startswith("http://localhost:8000/media/")

    submit_body = provider._client.post.call_args_list[2].kwargs["json"]
    assert submit_body["image_urls"] == [
        "https://fal.media/master.png",
        "https://fal.media/style.jpg",
    ]
    assert f"/{model}" in provider._client.post.call_args_list[2].args[0]
    # Turbo (the default) is distilled with a fixed step count and its
    # endpoint has no `num_inference_steps` input; only the standard model does.
    if model == flux_kontext.STANDARD_EDIT_MODEL:
        assert submit_body["num_inference_steps"] == NUM_INFERENCE_STEPS
    else:
        assert "num_inference_steps" not in submit_body
    assert submit_body["guidance_scale"] == GUIDANCE_SCALE
    assert submit_body["seed"] == 42
    assert "[IDENTITY]" not in submit_body["prompt"]


async def test_generate_caches_master_and_style_uploads_across_calls(
    storage: LocalFileStorage,
) -> None:
    """Real cost concern: re-uploading the same master/style reference on
    every candidate (or every panel in a batch using the same provider
    instance) would double-spend for nothing new — both must be uploaded
    at most once per provider instance."""
    provider = FluxKontextProvider(storage)
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, content=b"fake-master-bytes"),  # master fetch — once only
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"images": [{"url": "https://cdn.example/1.png"}]}),
            _response(200, content=b"fake-final-image"),
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"images": [{"url": "https://cdn.example/2.png"}]}),
            _response(200, content=b"fake-final-image-2"),
        ]
    )
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(
                200, {"file_url": "https://fal.media/master.png", "upload_url": "https://up/m"}
            ),
            _response(
                200, {"file_url": "https://fal.media/style.jpg", "upload_url": "https://up/s"}
            ),
            _response(
                200,
                {
                    "request_id": "req-1",
                    "status_url": "https://queue.fal.run/x/requests/req-1/status",
                    "response_url": "https://queue.fal.run/x/requests/req-1",
                },
            ),
            _response(
                200,
                {
                    "request_id": "req-2",
                    "status_url": "https://queue.fal.run/x/requests/req-2/status",
                    "response_url": "https://queue.fal.run/x/requests/req-2",
                },
            ),
        ]
    )
    provider._client.put = AsyncMock(return_value=_response(200))  # type: ignore[method-assign]

    await provider.generate(_prompt(panel_id=1), n=1, identity=_identity())
    await provider.generate(_prompt(panel_id=2), n=1, identity=_identity())

    # Only 3 POSTs total across both calls (2 uploads + 1st submit) plus
    # the 2nd call's own submit = 4; the upload POSTs (initiate) are not
    # repeated — call count proves the cache was hit the second time.
    assert provider._client.post.call_count == 4
    assert provider._client.get.call_count == 7  # 1 master fetch + 2x(status+result+image)


async def test_generate_raises_when_no_images_returned(storage: LocalFileStorage) -> None:
    provider = FluxKontextProvider(storage)
    provider._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, content=b"fake-master-bytes"),
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"images": []}),
        ]
    )
    provider._client.post = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(
                200, {"file_url": "https://fal.media/master.png", "upload_url": "https://up/m"}
            ),
            _response(
                200, {"file_url": "https://fal.media/style.jpg", "upload_url": "https://up/s"}
            ),
            _response(
                200,
                {
                    "request_id": "req-1",
                    "status_url": "https://queue.fal.run/x/requests/req-1/status",
                    "response_url": "https://queue.fal.run/x/requests/req-1",
                },
            ),
        ]
    )
    provider._client.put = AsyncMock(return_value=_response(200))  # type: ignore[method-assign]

    with pytest.raises(ImageGenerationError, match="panel 1"):
        await provider.generate(_prompt(), n=1, identity=_identity())


def test_estimate_usd_uses_output_size_only() -> None:
    # Provider built via __new__ to skip FAL_KEY/client setup.
    provider = FluxKontextProvider.__new__(FluxKontextProvider)
    assert provider.estimate_usd(n=4) == pytest.approx(OUTPUT_MEGAPIXELS * USD_PER_MEGAPIXEL * 4)


def test_turbo_is_the_default_and_a_third_cheaper_per_megapixel() -> None:
    """Turbo edit was chosen as the default after a real A/B against the
    standard edit model (`.data/model_compare/`)."""
    assert flux_kontext.MODEL_ID == flux_kontext.TURBO_EDIT_MODEL
    rates = flux_kontext.EDIT_MODEL_USD_PER_MEGAPIXEL
    assert rates[flux_kontext.TURBO_EDIT_MODEL] == 0.008
    saving = 1 - rates[flux_kontext.TURBO_EDIT_MODEL] / rates[flux_kontext.STANDARD_EDIT_MODEL]
    assert saving == pytest.approx(1 / 3)


def test_estimate_and_recorded_cost_follow_the_active_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(flux_kontext, "MODEL_ID", flux_kontext.STANDARD_EDIT_MODEL)
    provider = FluxKontextProvider.__new__(FluxKontextProvider)
    assert provider.estimate_usd(n=1) == pytest.approx(OUTPUT_MEGAPIXELS * 0.012)
