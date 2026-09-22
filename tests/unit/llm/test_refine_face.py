"""FaceRefiner — mocks the HTTP client and `detect_face_box` directly,
same pattern as `test_flux_kontext_provider.py`. Exercises the real
crop -> upscale -> re-edit -> feathered-composite path against canned
fal.ai responses. Face-bbox math itself (`face_bbox_from_kps`/
`detect_face_box`) lives in `ml/identity/faces.py`, shared with
`critic/identity.py` — tested in `test_faces.py`.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock

import httpx
import numpy as np
import pytest
from storage import LocalFileStorage

from app.images import refine_face
from app.images.refine_face import (
    MAX_REFINED_EDGE,
    MIN_REFINED_EDGE,
    UPSCALE_FACTOR,
    FaceRefiner,
    MissingFalConfigError,
    _feathered_composite,
    should_refine,
)
from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from app.prompts.style_card import load_style_card
from contracts import LookCard


def _look_card(**overrides: object) -> LookCard:
    fields: dict[str, object] = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "not enough information",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
    }
    fields.update(overrides)
    return LookCard(**fields)


def _response(
    status_code: int, json_data: dict[str, Any] | None = None, content: bytes = b""
) -> httpx.Response:
    request = httpx.Request("GET", "https://example.test")
    if json_data is not None:
        return httpx.Response(status_code, json=json_data, request=request)
    return httpx.Response(status_code, content=content, request=request)


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


# --- should_refine ---


def test_should_refine_true_for_wide_and_medium() -> None:
    assert should_refine("wide") is True
    assert should_refine("medium") is True


def test_should_refine_false_for_close_by_default() -> None:
    assert should_refine("close") is False


def test_should_refine_force_all_overrides_close() -> None:
    assert should_refine("close", force_all=True) is True


# --- feathered composite ---


def test_feathered_composite_uses_the_refined_crop_at_the_center() -> None:
    # A realistic face-crop-sized box (FEATHER_PX=24 is tuned against
    # crops this size, not a tiny test fixture) — deep interior pixels
    # should be fully refined, not still partway blended.
    panel = np.zeros((400, 400, 3), dtype=np.uint8)
    refined = np.full((200, 200, 3), 200, dtype=np.uint8)
    box = (100, 100, 300, 300)

    result = _feathered_composite(panel, refined, box)

    assert result[200, 200].tolist() == [200, 200, 200]  # deep inside the box: fully refined
    assert result[0, 0].tolist() == [0, 0, 0]  # far outside: untouched original


def test_feathered_composite_blends_at_the_edge_not_a_hard_seam() -> None:
    panel = np.zeros((400, 400, 3), dtype=np.uint8)
    refined = np.full((200, 200, 3), 255, dtype=np.uint8)
    box = (100, 100, 300, 300)

    result = _feathered_composite(panel, refined, box)
    edge_pixel = result[101, 200]  # just inside the box boundary
    # neither pure original (0) nor pure refined (255) — a real blend.
    assert 0 < int(edge_pixel[0]) < 255


# --- FaceRefiner.refine_panel (top-level, mocked fal calls) ---


def test_missing_api_key_raises(monkeypatch: pytest.MonkeyPatch, storage: LocalFileStorage) -> None:
    monkeypatch.delenv("FAL_KEY", raising=False)
    with pytest.raises(MissingFalConfigError, match="FAL_KEY"):
        FaceRefiner(storage)


async def test_refine_panel_skips_close_framing(storage: LocalFileStorage) -> None:
    refiner = FaceRefiner(storage)
    panel = np.zeros((200, 200, 3), dtype=np.uint8)

    result = await refiner.refine_panel(
        panel,
        framing="close",
        master_url="http://localhost:8000/media/people/me/master.png",
        look_card=_look_card(),
    )
    assert result is panel  # unchanged, no calls made


async def test_refine_panel_skips_when_no_face_detected(
    monkeypatch: pytest.MonkeyPatch, storage: LocalFileStorage
) -> None:
    monkeypatch.setattr(refine_face, "detect_face_box", lambda image_bgr: None)
    refiner = FaceRefiner(storage)
    panel = np.zeros((200, 200, 3), dtype=np.uint8)

    result = await refiner.refine_panel(
        panel,
        framing="wide",
        master_url="http://localhost:8000/media/people/me/master.png",
        look_card=_look_card(),
    )
    assert result is panel


async def test_refine_panel_happy_path(
    monkeypatch: pytest.MonkeyPatch, storage: LocalFileStorage
) -> None:
    monkeypatch.setattr(refine_face, "detect_face_box", lambda image_bgr: (85, 85, 115, 115))

    refiner = FaceRefiner(storage)
    # Call order: GET master bytes -> POST+PUT upload master -> POST+PUT
    # upload style ref -> POST+PUT upload crop -> POST submit -> GET
    # status -> GET result -> GET final image bytes.
    refiner._client.get = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(200, content=b"fake-master-bytes"),
            _response(200, {"status": "COMPLETED"}),
            _response(200, {"images": [{"url": "https://cdn.example/refined.png"}]}),
            _response(200, content=b"fake-refined-bytes"),
        ]
    )
    refiner._client.post = AsyncMock(  # type: ignore[method-assign]
        side_effect=[
            _response(
                200, {"file_url": "https://fal.media/master.png", "upload_url": "https://up/m"}
            ),
            _response(
                200, {"file_url": "https://fal.media/style.jpg", "upload_url": "https://up/s"}
            ),
            _response(
                200, {"file_url": "https://fal.media/crop.jpg", "upload_url": "https://up/c"}
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
    refiner._client.put = AsyncMock(return_value=_response(200))  # type: ignore[method-assign]

    # decode_jpeg_bgr is used inside refine_crop to decode the downloaded
    # "refined" bytes — patch it to avoid needing a real JPEG payload.
    monkeypatch.setattr(
        refine_face, "decode_jpeg_bgr", lambda data: np.full((240, 240, 3), 128, dtype=np.uint8)
    )

    panel = np.zeros((200, 200, 3), dtype=np.uint8)
    style = load_style_card()
    result = await refiner.refine_panel(
        panel,
        framing="wide",
        master_url="http://localhost:8000/media/people/me/master.png",
        look_card=_look_card(),
        style=style,
    )

    assert result.shape == panel.shape
    # the face region should no longer be all-zero — something was composited in.
    assert result[100, 100].sum() > 0

    submit_body = refiner._client.post.call_args_list[3].kwargs["json"]
    assert len(submit_body["image_urls"]) == 3  # crop, master, style
    assert submit_body["image_size"]["width"] >= MIN_REFINED_EDGE
    assert submit_body["image_size"]["width"] <= MAX_REFINED_EDGE


def test_upscale_factor_is_applied_within_clamped_bounds() -> None:
    # A tiny crop should clamp up to MIN_REFINED_EDGE, not stay tiny.
    small_target = max(MIN_REFINED_EDGE, min(MAX_REFINED_EDGE, int(20 * UPSCALE_FACTOR)))
    assert small_target == MIN_REFINED_EDGE
    # A huge crop should clamp down to MAX_REFINED_EDGE, not grow unbounded.
    huge_target = max(MIN_REFINED_EDGE, min(MAX_REFINED_EDGE, int(2000 * UPSCALE_FACTOR)))
    assert huge_target == MAX_REFINED_EDGE
