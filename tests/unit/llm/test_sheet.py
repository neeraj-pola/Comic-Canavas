"""ml/identity/sheet.py — the character sheet pipeline. Every fal.ai
call (`upload_image`/`submit_and_wait`/`download`) and every
scoring call (`detect_and_embed`/`score_checklist`/`identity_similarity`/
`master_face_embedding`) is monkeypatched at the module level; these
tests never touch the network or load a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.prompts.style_card import load_style_card
from contracts import FeatureCheck, FeatureChecklistResult, LookCard, SheetImage
from ml.identity import sheet
from ml.identity.faces import DetectedFaceResult
from ml.identity.model import IdentityModel, JsonIdentityModelStore


def _look_card(**overrides: object) -> LookCard:
    fields: dict[str, object] = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "not enough information",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
        "age_descriptor": "24-year-old",
    }
    fields.update(overrides)
    return LookCard(**fields)


def _checklist(
    *, mandatory_passed: bool = True, artifacts_clean: bool = True
) -> FeatureChecklistResult:
    return FeatureChecklistResult(
        checks=[FeatureCheck(feature="hair", expected="x", present=True, mandatory=True)],
        score=1.0 if mandatory_passed else 0.5,
        mandatory_passed=mandatory_passed,
        artifacts_clean=artifacts_clean,
        image_sha256="x",
        model="mock:mock",
    )


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


async def _async_result(value: object) -> object:
    return value


# --- build_sheet_prompt ---


def test_build_sheet_prompt_includes_style_phrase_traits_view_and_expression() -> None:
    style = load_style_card()
    prompt = sheet.build_sheet_prompt(
        _look_card(), style, view=sheet.VIEWS[1], expression="a warm smile"
    )
    assert style.style_phrase in prompt
    assert "24-year-old man" in prompt
    assert "short black wavy hair" in prompt
    assert sheet.VIEWS[1] in prompt
    assert "a warm smile" in prompt


def test_build_sheet_prompt_includes_the_expression_dropout_guard() -> None:
    """The prompt must guard against expression-driven feature dropout —
    the same failure mode the editing model shows elsewhere under a
    strained or exaggerated expression."""
    style = load_style_card()
    prompt = sheet.build_sheet_prompt(_look_card(), style, view=sheet.VIEWS[0], expression="x")
    assert "EVEN under a strained or exaggerated expression" in prompt


# --- VIEWS/EXPRESSIONS balance ---


def test_views_has_three_explicitly_left_and_right_entries() -> None:
    assert len(sheet.VIEWS) == 3
    joined = " ".join(sheet.VIEWS).lower()
    assert "left" in joined
    assert "right" in joined


# --- generate_sheet_images ---


async def test_generate_sheet_images_produces_unique_ids_for_the_full_sheet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`view_index = i % len(VIEWS)` paired with `expression_index = i %
    len(EXPRESSIONS)` would only produce `len(EXPRESSIONS)` unique ids
    when one length evenly divides the other (e.g. 3 into 12), silently
    colliding in `images_by_id` before ever reaching `persist_sheet`.
    Every id across the full `MAX_SHEET` must be unique."""
    monkeypatch.setattr(sheet, "upload_image", lambda *_a, **_k: _async_result("master-url"))
    monkeypatch.setattr(
        sheet,
        "submit_and_wait",
        lambda *_a, **_k: _async_result({"images": [{"url": "https://cdn.example/x.png"}]}),
    )
    monkeypatch.setattr(sheet, "download", lambda *_a, **_k: _async_result(b"fake-image-bytes"))

    request = sheet.SheetRequest(person_id="me", master_jpeg=b"fake-master", look_card=_look_card())
    style = load_style_card()
    raws = await sheet.generate_sheet_images(request, style, n=sheet.MAX_SHEET, client=object())  # type: ignore[arg-type]

    assert len(raws) == sheet.MAX_SHEET
    ids = [r[0] for r in raws]
    assert len(set(ids)) == sheet.MAX_SHEET  # zero collisions — the actual bug this catches

    views_used = {view for _id, _data, view, _expr in raws}
    assert views_used == set(sheet.VIEWS)  # all three views appear
    # each view gets exactly len(EXPRESSIONS) images — even coverage
    for view in sheet.VIEWS:
        assert sum(1 for _id, _data, v, _expr in raws if v == view) == len(sheet.EXPRESSIONS)


async def test_generate_sheet_images_rejects_n_larger_than_max_sheet() -> None:
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"fake-master", look_card=_look_card())
    style = load_style_card()
    with pytest.raises(ValueError, match="MAX_SHEET"):
        await sheet.generate_sheet_images(
            request,
            style,
            n=sheet.MAX_SHEET + 1,
            client=object(),  # type: ignore[arg-type]
        )


async def test_generate_sheet_images_uploads_master_once(monkeypatch: pytest.MonkeyPatch) -> None:
    upload_calls: list[tuple[bytes, str, str]] = []

    async def fake_upload_image(
        data: bytes, *, client: object, file_name: str, content_type: str
    ) -> str:
        upload_calls.append((data, file_name, content_type))
        return "master-url"

    monkeypatch.setattr(sheet, "upload_image", fake_upload_image)
    monkeypatch.setattr(
        sheet,
        "submit_and_wait",
        lambda *_a, **_k: _async_result({"images": [{"url": "https://cdn.example/x.png"}]}),
    )
    monkeypatch.setattr(sheet, "download", lambda *_a, **_k: _async_result(b"fake-image-bytes"))

    request = sheet.SheetRequest(person_id="me", master_jpeg=b"fake-master", look_card=_look_card())
    style = load_style_card()
    await sheet.generate_sheet_images(request, style, n=4, client=object())  # type: ignore[arg-type]

    assert len(upload_calls) == 1  # not once per image
    assert upload_calls[0][0] == b"fake-master"
    assert upload_calls[0][2] == "image/jpeg"


# --- score_sheet / select_kept ---


async def test_score_sheet_keeps_a_candidate_above_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=np.ones(512, np.float32), face_count=1
    )
    monkeypatch.setattr(sheet, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(sheet, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(
        sheet, "score_checklist", lambda *_a, **_k: _async_result(_checklist(mandatory_passed=True))
    )
    monkeypatch.setattr(sheet, "identity_similarity", lambda *_a, **_k: 0.8)

    raws = [("0-0", b"fake-jpeg", sheet.VIEWS[0], "a warm smile")]
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"master", look_card=_look_card())
    scored = await sheet.score_sheet(raws, request, master_embedding=_unit([1.0, 0.0]))

    assert scored[0].kept is True
    assert scored[0].dinov2_to_master == 0.8
    assert sheet.select_kept(scored) == scored


async def test_score_sheet_drops_a_candidate_below_dinov2_threshold(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=np.ones(512, np.float32), face_count=1
    )
    monkeypatch.setattr(sheet, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(sheet, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(
        sheet, "score_checklist", lambda *_a, **_k: _async_result(_checklist(mandatory_passed=True))
    )
    monkeypatch.setattr(sheet, "identity_similarity", lambda *_a, **_k: 0.5)  # < 0.7 threshold

    raws = [("0-0", b"fake-jpeg", sheet.VIEWS[0], "a warm smile")]
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"master", look_card=_look_card())
    scored = await sheet.score_sheet(raws, request, master_embedding=_unit([1.0, 0.0]))

    assert scored[0].kept is False
    assert sheet.select_kept(scored) == []


async def test_score_sheet_drops_a_candidate_failing_the_checklist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=np.ones(512, np.float32), face_count=1
    )
    monkeypatch.setattr(sheet, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(sheet, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(
        sheet,
        "score_checklist",
        lambda *_a, **_k: _async_result(_checklist(mandatory_passed=False)),
    )
    monkeypatch.setattr(sheet, "identity_similarity", lambda *_a, **_k: 0.9)

    raws = [("0-0", b"fake-jpeg", sheet.VIEWS[0], "a warm smile")]
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"master", look_card=_look_card())
    scored = await sheet.score_sheet(raws, request, master_embedding=_unit([1.0, 0.0]))

    assert scored[0].kept is False


async def test_score_sheet_treats_no_detected_face_as_zero_dinov2(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`identity_similarity` returns None when no face is detected in the
    candidate (InsightFace limitation on stylized art) — score_sheet must
    not crash, and must not keep a candidate it can't actually measure."""
    monkeypatch.setattr(sheet, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(sheet, "detect_and_embed", lambda image_bgr: None)
    monkeypatch.setattr(
        sheet, "score_checklist", lambda *_a, **_k: _async_result(_checklist(mandatory_passed=True))
    )
    monkeypatch.setattr(sheet, "identity_similarity", lambda *_a, **_k: None)

    raws = [("0-0", b"fake-jpeg", sheet.VIEWS[0], "a warm smile")]
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"master", look_card=_look_card())
    scored = await sheet.score_sheet(raws, request, master_embedding=_unit([1.0, 0.0]))

    assert scored[0].dinov2_to_master == 0.0
    assert scored[0].kept is False


# --- design_sheet ---


async def test_design_sheet_raises_without_confirm() -> None:
    request = sheet.SheetRequest(person_id="me", master_jpeg=b"master", look_card=_look_card())
    with pytest.raises(sheet.SpendNotConfirmedError):
        await sheet.design_sheet(request, confirm=False)


# --- persist_sheet ---


class _FakeStorage:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_presigned(self, key: str, **kwargs: object) -> str:
        raise NotImplementedError

    def get_url(self, key: str) -> str:
        return f"https://example.test/{key}"

    def get_object_by_url(self, url: str) -> bytes:
        prefix = "https://example.test/"
        return self.objects[url[len(prefix) :]]

    def put_object(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        self.objects[key] = data
        return self.get_url(key)

    def delete_prefix(self, prefix: str) -> None:
        raise NotImplementedError


def _sheet_image(image_id: str) -> SheetImage:
    return SheetImage(
        id=image_id,
        url="",
        view=sheet.VIEWS[0],
        expression="a warm smile",
        checklist=_checklist(),
        dinov2_to_master=0.8,
        kept=True,
    )


def test_persist_sheet_writes_every_kept_image_and_records_paths(tmp_path: Path) -> None:
    storage = _FakeStorage()
    store = JsonIdentityModelStore(tmp_path)
    kept = [_sheet_image("0-0"), _sheet_image("1-1")]
    images_by_id = {"0-0": b"image-a", "1-1": b"image-b"}

    result = sheet.persist_sheet(store, "me", kept, images_by_id, storage=storage)

    assert storage.objects["people/me/sheet/0-0.png"] == b"image-a"
    assert storage.objects["people/me/sheet/1-1.png"] == b"image-b"
    assert len(result.sheet_paths) == 2
    assert result.sheet_generated_at is not None

    reloaded = store.load("me")
    assert reloaded is not None
    assert reloaded.sheet_paths == result.sheet_paths


def test_persist_sheet_preserves_existing_identity_fields(tmp_path: Path) -> None:
    store = JsonIdentityModelStore(tmp_path)
    store.save(IdentityModel(person_id="me", trigger_token="zkqid"))

    storage = _FakeStorage()
    sheet.persist_sheet(store, "me", [_sheet_image("0-0")], {"0-0": b"image-a"}, storage=storage)

    reloaded = store.load("me")
    assert reloaded is not None
    assert reloaded.trigger_token == "zkqid"
    assert len(reloaded.sheet_paths) == 1
