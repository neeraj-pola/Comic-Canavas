"""ml/identity/lookcard.py — selecting the 6 sharpest usable crops and
the vision call that turns them into a LookCard. Uses the mock LLM
provider (LLM_LOOKCARD routed to mock), same pattern as every other
routed-role test — live-verified separately against both real providers
(openai:gpt-4o and anthropic:claude-sonnet-4-6).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.llm import routing
from app.llm.mock_provider import MockProvider
from contracts import LookCard
from ml.identity.faces import FacePhoto, PhotoStatus
from ml.identity.lookcard import (
    MAX_CROPS,
    NoUsablePhotosError,
    build_look_card,
    select_best_crops,
)


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("lookcard")
    assert isinstance(provider, MockProvider)
    return provider


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_LOOKCARD", "mock:mock-lookcard")


def _card_json(**overrides: str) -> str:
    defaults = {
        "hair": "short black hair",
        "glasses": "none",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "not enough information",
        "distinguishing": "none noted",
    }
    defaults.update(overrides)
    return json.dumps(defaults)


def _photo(status: PhotoStatus, blur_score: float | None) -> FacePhoto:
    has_face = status not in ("no_face", "multi_face")
    crop = np.zeros((4, 4, 3), dtype=np.uint8) if has_face else None
    return FacePhoto(
        path=Path(f"/x/{status}-{blur_score}.jpg"),
        status=status,
        blur_score=blur_score,
        crop_bgr=crop,
    )


def test_select_best_crops_picks_sharpest_usable_photos_first() -> None:
    photos = [
        _photo("usable", 100.0),
        _photo("usable", 500.0),
        _photo("blur", 10.0),
        _photo("usable", 300.0),
        _photo("no_face", None),
    ]
    crops = select_best_crops(photos, n=2)
    assert len(crops) == 2  # sharpest two usable photos (500, 300), blur/no_face excluded


def test_select_best_crops_caps_at_max_crops_default() -> None:
    photos = [_photo("usable", float(i)) for i in range(10)]
    crops = select_best_crops(photos)
    assert len(crops) == MAX_CROPS


def test_select_best_crops_raises_when_nothing_usable() -> None:
    photos = [_photo("blur", 10.0), _photo("no_face", None)]
    with pytest.raises(NoUsablePhotosError):
        select_best_crops(photos)


async def test_build_look_card_returns_a_valid_card() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionLookCardFields", _card_json(glasses="dark rectangular frames")
    )

    card = await build_look_card([b"fake-jpeg-bytes"])

    assert isinstance(card, LookCard)
    assert card.glasses == "dark rectangular frames"
    assert card.gender_term == ""  # never asked of the vision call — see module docstring


async def test_build_look_card_passes_through_caller_supplied_gender_term() -> None:
    """gender_term is never inferred by the vision call (see this
    module's docstring and `ml/identity/lookcard.py`'s) — it's whatever
    the caller passes, verbatim."""
    provider = _resolved_mock_provider()
    provider.set_structured_response("_VisionLookCardFields", _card_json())

    card = await build_look_card([b"fake-jpeg-bytes"], gender_term="woman")

    assert card.gender_term == "woman"


async def test_build_look_card_raises_on_empty_crop_list() -> None:
    with pytest.raises(NoUsablePhotosError):
        await build_look_card([])
