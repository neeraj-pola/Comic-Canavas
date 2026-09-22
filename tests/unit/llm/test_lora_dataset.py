"""ml/identity/lora_dataset.py — selecting training photos, captioning
them (with the refusal-fallback), and writing the ai-toolkit-format
dataset folder. Uses the mock LLM provider; a Claude fallback handles
photos an initial provider refuses to caption (see the module's own
docstring).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from app.llm import routing
from app.llm.base import LLMResult
from app.llm.mock_provider import MockProvider
from contracts import FeatureCheck, FeatureChecklistResult, LookCard, SheetImage
from ml.identity.faces import FacePhoto
from ml.identity.lora_dataset import (
    BACKGROUND_MASK_VALUE,
    CHARACTER_TRIGGER_TOKEN,
    FACE_MASK_VALUE,
    MIN_TRAINING_PHOTOS,
    TooFewTrainingPhotosError,
    _face_region_mask,
    build_character_dataset,
    build_training_dataset,
    caption_training_photo,
    select_training_photos,
)


def _look_card(**overrides: str) -> LookCard:
    fields = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "mustard sweater",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
    }
    fields.update(overrides)
    return LookCard(**fields)


def _photo(blur_score: float) -> FacePhoto:
    import numpy as np

    return FacePhoto(
        path=Path(f"/x/{blur_score}.jpg"),
        status="usable",
        blur_score=blur_score,
        crop_bgr=np.zeros((4, 4, 3), dtype=np.uint8),
    )


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_LOOKCARD", "mock:mock-lookcard")


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("lookcard")
    assert isinstance(provider, MockProvider)
    return provider


def test_select_training_photos_sorts_by_blur_score_descending() -> None:
    photos = [_photo(100.0), _photo(500.0), _photo(300.0)]
    selected = select_training_photos(photos, n=2)
    assert [p.blur_score for p in selected] == [500.0, 300.0]


async def test_caption_training_photo_returns_a_normal_caption() -> None:
    provider = _resolved_mock_provider()
    provider.complete_text = "outdoors near a lake, wearing a jacket"

    caption = await caption_training_photo(b"fake-jpeg-bytes")

    assert caption == "outdoors near a lake, wearing a jacket"


async def test_caption_training_photo_falls_back_when_primary_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _resolved_mock_provider()
    provider.complete_text = "I'm sorry, I can't help with that."

    fake_result = LLMResult(
        text="outdoors, wearing a jacket",
        input_tokens=10,
        output_tokens=5,
        usd=0.0,
        provider="anthropic",
        model="claude-sonnet-4-6",
        latency_ms=1,
    )
    fake_fallback = AsyncMock()
    fake_fallback.complete = AsyncMock(return_value=fake_result)
    monkeypatch.setattr("app.llm.anthropic_provider.AnthropicProvider", lambda: fake_fallback)

    caption = await caption_training_photo(b"fake-jpeg-bytes")

    assert caption == "outdoors, wearing a jacket"
    fake_fallback.complete.assert_awaited_once()


async def test_build_training_dataset_writes_image_and_caption_pairs(tmp_path: Path) -> None:
    provider = _resolved_mock_provider()
    provider.complete_text = "outdoors, wearing a jacket"

    photos = [_photo(float(i * 100)) for i in range(1, MIN_TRAINING_PHOTOS + 1)]
    output_dir = tmp_path / "dataset"

    count = await build_training_dataset(photos, output_dir, trigger_token="zkqid")

    assert count == MIN_TRAINING_PHOTOS
    assert len(list(output_dir.glob("*.jpg"))) == MIN_TRAINING_PHOTOS
    assert len(list(output_dir.glob("*.txt"))) == MIN_TRAINING_PHOTOS
    caption = (output_dir / "0.txt").read_text()
    assert caption == "zkqid a photo of a person, outdoors, wearing a jacket"


async def test_build_training_dataset_anchors_caption_with_gender_and_look_card(
    tmp_path: Path,
) -> None:
    """The fixed caption prefix carries gender + stable traits (hair,
    glasses, distinguishing) so the trigger token anchors to the right
    class prior instead of an ambiguous "person" — see the module
    docstring for why this doesn't reintroduce the disentanglement
    problem `trainingcaption.v1.md` warns about (this prefix, like the
    trigger token, is constant across every caption; only the per-photo
    description below still varies)."""
    provider = _resolved_mock_provider()
    provider.complete_text = "outdoors, wearing a jacket"

    photos = [_photo(float(i * 100)) for i in range(1, MIN_TRAINING_PHOTOS + 1)]
    output_dir = tmp_path / "dataset"

    await build_training_dataset(
        photos,
        output_dir,
        trigger_token="zkqid",
        gender_term="man",
        look_card=_look_card(),
    )

    caption = (output_dir / "0.txt").read_text()
    assert caption == (
        "zkqid a photo of a man, short black wavy hair, black rectangular glasses, "
        "a small mole above the left eyebrow, outdoors, wearing a jacket"
    )


async def test_build_training_dataset_omits_glasses_when_none(tmp_path: Path) -> None:
    provider = _resolved_mock_provider()
    provider.complete_text = "outdoors, wearing a jacket"

    photos = [_photo(float(i * 100)) for i in range(1, MIN_TRAINING_PHOTOS + 1)]
    output_dir = tmp_path / "dataset"

    await build_training_dataset(
        photos,
        output_dir,
        trigger_token="zkqid",
        gender_term="woman",
        look_card=_look_card(glasses="none", distinguishing="none noted"),
    )

    caption = (output_dir / "0.txt").read_text()
    assert "glasses" not in caption
    assert caption == "zkqid a photo of a woman, short black wavy hair, outdoors, wearing a jacket"


async def test_build_training_dataset_includes_age_descriptor_when_set(tmp_path: Path) -> None:
    provider = _resolved_mock_provider()
    provider.complete_text = "outdoors, wearing a jacket"

    photos = [_photo(float(i * 100)) for i in range(1, MIN_TRAINING_PHOTOS + 1)]
    output_dir = tmp_path / "dataset"

    await build_training_dataset(
        photos,
        output_dir,
        trigger_token="zkqid",
        gender_term="man",
        look_card=_look_card(age_descriptor="24-year-old"),
    )

    caption = (output_dir / "0.txt").read_text()
    assert caption.startswith("zkqid a photo of a 24-year-old man, ")


async def test_build_training_dataset_raises_when_too_few_photos(tmp_path: Path) -> None:
    photos = [_photo(100.0)] * (MIN_TRAINING_PHOTOS - 1)
    with pytest.raises(TooFewTrainingPhotosError):
        await build_training_dataset(photos, tmp_path / "dataset", trigger_token="zkqid")


# --- v2: character-sheet dataset -------------------------------------------


def _fake_jpeg(size: int = 64) -> bytes:
    from ml.identity.faces import encode_crop_jpeg

    return encode_crop_jpeg(np.full((size, size, 3), 128, dtype=np.uint8))


def _sheet_image(
    image_id: str, *, view: str = "facing directly forward", expression: str = "a warm smile"
) -> SheetImage:
    return SheetImage(
        id=image_id,
        url="",
        view=view,
        expression=expression,
        checklist=FeatureChecklistResult(
            checks=[FeatureCheck(feature="hair", expected="x", present=True, mandatory=True)],
            score=1.0,
            mandatory_passed=True,
            artifacts_clean=True,
            image_sha256="x",
            model="mock:mock",
        ),
        dinov2_to_master=0.85,
        kept=True,
    )


def test_face_region_mask_has_a_higher_value_face_region_than_background() -> None:
    # A face occupying a small fraction of a much larger frame — like a
    # real 1024x1024 training image, not a tight aligned crop.
    kps = np.array([[145, 145], [165, 145], [155, 155], [147, 165], [163, 165]], dtype=np.float32)
    mask = _face_region_mask((300, 300), kps)

    assert mask[155, 155] == FACE_MASK_VALUE  # face center
    assert mask[5, 5] == BACKGROUND_MASK_VALUE  # far corner, well outside the ellipse
    assert pytest.approx(2.5, abs=0.05) == FACE_MASK_VALUE / BACKGROUND_MASK_VALUE


def test_face_region_mask_is_all_background_without_a_detected_face() -> None:
    mask = _face_region_mask((32, 32), None)
    assert (mask == BACKGROUND_MASK_VALUE).all()


def test_build_character_dataset_writes_images_captions_and_masks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import ml.identity.lora_dataset as lora_dataset_module

    monkeypatch.setattr(
        lora_dataset_module, "largest_face_kps", lambda image_bgr: None
    )  # no real model load in tests

    sheet_images = [
        _sheet_image("0-0", view="facing directly forward", expression="a warm smile"),
        _sheet_image(
            "1-4", view="turned three-quarters to their own left", expression="an excited grin"
        ),
    ]
    images_by_id = {"0-0": _fake_jpeg(), "1-4": _fake_jpeg()}
    output_dir = tmp_path / "character_dataset"

    count = build_character_dataset(sheet_images, images_by_id, output_dir)

    assert count == 2
    assert (output_dir / "0.jpg").exists()
    assert (output_dir / "1.jpg").exists()
    assert (output_dir / "masks" / "0.png").exists()
    assert (output_dir / "masks" / "1.png").exists()

    caption0 = (output_dir / "0.txt").read_text()
    assert caption0 == (
        f"{CHARACTER_TRIGGER_TOKEN}, comic character, facing directly forward, "
        "a warm smile, a black t-shirt, a plain cream background"
    )
    # no identity words (hair/glasses/etc.) anywhere in the caption — the
    # actual point of this captioning convention.
    for banned in ("hair", "glasses", "mustache", "mole"):
        assert banned not in caption0

    caption1 = (output_dir / "1.txt").read_text()
    assert "turned three-quarters to their own left" in caption1
    assert "an excited grin" in caption1
