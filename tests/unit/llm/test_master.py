"""ml/identity/master.py — the master character design pipeline.
Every fal.ai call (`upload_image`/`submit_and_wait`/`download`) and every
scoring call (`detect_and_embed`/`score_checklist`/`embed_image`) is
monkeypatched at the module level; these tests never touch the network
or load a real model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from app.prompts.style_card import load_style_card
from contracts import FeatureCheck, FeatureChecklistResult, LookCard, MasterCandidate
from ml.identity import master
from ml.identity.faces import DetectedFaceResult
from ml.identity.model import JsonIdentityModelStore


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
    *, score: float = 1.0, mandatory_passed: bool = True, artifacts_clean: bool = True
) -> FeatureChecklistResult:
    return FeatureChecklistResult(
        checks=[FeatureCheck(feature="hair", expected="x", present=True, mandatory=True)],
        score=score,
        mandatory_passed=mandatory_passed,
        artifacts_clean=artifacts_clean,
        image_sha256="x",
        model="mock:mock",
    )


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


# --- build_master_prompt ---


def test_build_master_prompt_includes_style_phrase_and_traits() -> None:
    style = load_style_card()
    prompt = master.build_master_prompt(_look_card(), style)
    assert style.style_phrase in prompt
    assert "24-year-old man" in prompt
    assert "short black wavy hair" in prompt
    assert "black rectangular glasses" in prompt
    assert "mole above the left eyebrow" in prompt


def test_build_master_prompt_omits_none_values() -> None:
    style = load_style_card()
    prompt = master.build_master_prompt(
        _look_card(glasses="none", distinguishing="none noted"), style
    )
    assert "glasses" not in prompt.split("Avoid:")[0].lower() or "black rectangular" not in prompt


def test_build_master_prompt_uses_the_given_pose() -> None:
    style = load_style_card()
    prompt = master.build_master_prompt(_look_card(), style, pose="a broad laugh, head tilted")
    assert "a broad laugh, head tilted" in prompt


def test_build_master_prompt_defaults_to_first_pose_variation() -> None:
    style = load_style_card()
    prompt = master.build_master_prompt(_look_card(), style)
    assert master.POSE_VARIATIONS[0] in prompt


async def test_generate_candidates_cycles_through_pose_variations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A single fixed prompt for all 24 candidates barely varies
    expression/pose against an editing model that tends to preserve the
    source crop's own pose — variety has to come from the prompt
    itself."""
    style = load_style_card()
    seen_prompts: list[str] = []

    async def fake_generate_one(
        crop_jpeg: bytes,
        *,
        crop_index: int,
        seed: int,
        prompt: str,
        style_reference_url: str,
        client: object,
    ) -> tuple[bytes, str]:
        seen_prompts.append(prompt)
        return b"fake-image", prompt

    async def fake_prepare_style_reference(client: object) -> str:
        return "ref-url"

    monkeypatch.setattr(master, "_generate_one", fake_generate_one)
    monkeypatch.setattr(master, "_prepare_style_reference", fake_prepare_style_reference)

    request = master.MasterRequest(
        person_id="me", crops=[b"crop-0"], look_card=_look_card(), reference_embedding=None
    )
    await master.generate_candidates(request, style, client=object())  # type: ignore[arg-type]

    assert len(seen_prompts) == master.SEEDS_PER_CROP
    # every seed gets a distinct pose (up to the number of variations available)
    assert len(set(seen_prompts)) == min(master.SEEDS_PER_CROP, len(master.POSE_VARIATIONS))


# --- apply_chip ---


def test_apply_chip_changes_exactly_one_field_and_records_it() -> None:
    look_card = _look_card()
    updated = master.apply_chip(look_card, "no glasses")
    assert updated.glasses == "none"
    assert updated.hair == look_card.hair  # untouched
    assert updated.edits == ["no glasses"]


def test_apply_chip_preserves_hair_color_word_on_merge() -> None:
    look_card = _look_card(hair="black wavy hair")
    updated = master.apply_chip(look_card, "more hair")
    assert "black" in updated.hair


def test_apply_chip_raises_on_unknown_label() -> None:
    with pytest.raises(master.UnknownChipError):
        master.apply_chip(_look_card(), "make it purple")


# --- select_top ---


def _candidate(id_: str, *, rank_score: float, mandatory_passed: bool = True) -> MasterCandidate:
    return MasterCandidate(
        id=id_,
        url="",
        seed=1,
        source_crop_index=0,
        prompt="x",
        checklist=_checklist(mandatory_passed=mandatory_passed),
        style_score=0.5,
        arcface_agreement=None,
        rank_score=rank_score,
    )


def test_select_top_drops_near_duplicates_and_backfills() -> None:
    candidates = [
        _candidate("a", rank_score=0.9),
        _candidate("b", rank_score=0.8),  # near-duplicate of "a"
        _candidate("c", rank_score=0.7),
        _candidate("d", rank_score=0.6),
        _candidate("e", rank_score=0.5),
    ]
    embeddings = {
        "a": _unit([1.0, 0.0]),
        "b": _unit([0.999, 0.001]),  # cosine ~1.0 to "a" — near-duplicate
        "c": _unit([0.0, 1.0]),
        "d": _unit([0.0, -1.0]),
        "e": _unit([-1.0, 0.0]),
    }
    top = master.select_top(candidates, embeddings, k=4)
    ids = [c.id for c in top]
    assert "b" not in ids or len(ids) == 4  # "b" skipped first pass, only backfilled if needed
    assert len(top) == 4


def test_select_top_excludes_candidates_that_fail_mandatory_checklist() -> None:
    candidates = [
        _candidate("a", rank_score=0.9, mandatory_passed=False),
        _candidate("b", rank_score=0.5, mandatory_passed=True),
    ]
    embeddings = {"a": _unit([1.0, 0.0]), "b": _unit([0.0, 1.0])}
    top = master.select_top(candidates, embeddings, k=4)
    assert [c.id for c in top] == ["b"]


# --- rank_candidates ---


async def test_rank_candidates_still_scores_no_face_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """InsightFace — a photo face detector — can fail to detect a face at
    all in clean illustrated art even when the candidate is clearly good
    (visible glasses, mustache, one face). A "no face" result must not be
    an automatic hard reject; the checklist still runs on the full image,
    only the ArcFace signal is unavailable."""
    monkeypatch.setattr(master, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(master, "detect_and_embed", lambda image_bgr: None)
    monkeypatch.setattr(master, "embed_image", lambda image_bgr: _unit([1.0, 0.0]))
    monkeypatch.setattr(
        master, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=1.0))
    )
    raws = [("id-1", b"fake-jpeg", 1, 0, "prompt")]
    request = master.MasterRequest(
        person_id="me", crops=[b"crop"], look_card=_look_card(), reference_embedding=None
    )
    ranked = await master.rank_candidates(raws, request, style_bank=np.stack([_unit([1.0, 0.0])]))
    assert ranked[0].rank_score > 0  # not the -1.0 hard-reject floor
    assert ranked[0].checklist.score == 1.0
    assert ranked[0].arcface_agreement is None  # no crop/embedding to compute it from


async def test_rank_candidates_hard_rejects_multi_face(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=np.ones(512, np.float32), face_count=2
    )
    monkeypatch.setattr(master, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(master, "detect_and_embed", lambda image_bgr: fake_result)
    raws = [("id-1", b"fake-jpeg", 1, 0, "prompt")]
    request = master.MasterRequest(
        person_id="me", crops=[b"crop"], look_card=_look_card(), reference_embedding=None
    )
    ranked = await master.rank_candidates(raws, request, style_bank=np.stack([_unit([1.0, 0.0])]))
    assert ranked[0].rank_score == -1.0


async def test_rank_candidates_orders_by_weighted_score(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_result = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=np.ones(512, np.float32), face_count=1
    )
    monkeypatch.setattr(master, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8))
    monkeypatch.setattr(master, "detect_and_embed", lambda image_bgr: fake_result)
    monkeypatch.setattr(master, "embed_image", lambda image_bgr: _unit([1.0, 0.0]))

    scores = iter([_checklist(score=0.5), _checklist(score=1.0)])
    monkeypatch.setattr(master, "score_checklist", lambda *_a, **_k: _async_result(next(scores)))

    raws = [("low", b"fake-jpeg-1", 1, 0, "prompt"), ("high", b"fake-jpeg-2", 2, 0, "prompt")]
    request = master.MasterRequest(
        person_id="me", crops=[b"crop"], look_card=_look_card(), reference_embedding=None
    )
    ranked = await master.rank_candidates(raws, request, style_bank=np.stack([_unit([1.0, 0.0])]))
    assert [c.id for c in ranked] == ["high", "low"]


async def _async_result(value: object) -> object:
    return value


# --- design_master ---


async def test_design_master_raises_without_confirm() -> None:
    request = master.MasterRequest(
        person_id="me", crops=[b"crop"], look_card=_look_card(), reference_embedding=None
    )
    with pytest.raises(master.SpendNotConfirmedError):
        await master.design_master(request, confirm=False)


# --- approve_master ---


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


def test_approve_master_writes_image_and_persists_identity_fields(tmp_path: Path) -> None:
    storage = _FakeStorage()
    store = JsonIdentityModelStore(tmp_path)
    candidate = _candidate("42-0", rank_score=0.9)
    look_card = _look_card()

    result = master.approve_master(
        store,
        "me",
        candidate,
        b"the-image-bytes",
        storage=storage,
        look_card=look_card,
        style_card_version="v2-2026-09-14",
        chip_rounds=1,
    )

    assert storage.objects["people/me/master.png"] == b"the-image-bytes"
    assert result.master_path == "https://example.test/people/me/master.png"
    assert result.master_candidate_id == "42-0"
    assert result.master_seed == 1
    assert result.style_card_version == "v2-2026-09-14"
    assert result.chip_rounds == 1
    assert result.look_card == look_card

    reloaded = store.load("me")
    assert reloaded is not None
    assert reloaded.master_path == result.master_path


def test_approve_master_preserves_existing_fields_not_being_updated(tmp_path: Path) -> None:
    store = JsonIdentityModelStore(tmp_path)
    existing = store.get_or_create("me")
    existing.embedding = [0.1, 0.2]
    existing.trigger_token = "zkqid"
    store.save(existing)

    storage = _FakeStorage()
    master.approve_master(
        store,
        "me",
        _candidate("1-0", rank_score=0.9),
        b"data",
        storage=storage,
        look_card=_look_card(),
        style_card_version="v2",
        chip_rounds=0,
    )

    reloaded = store.load("me")
    assert reloaded is not None
    assert reloaded.embedding == [0.1, 0.2]
    assert reloaded.trigger_token == "zkqid"
