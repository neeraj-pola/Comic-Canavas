"""critic/identity.py — the identity scoring signal for per-panel
candidate selection. Every scoring call (`detect_and_embed`/
`score_checklist`/`detect_face_box`) is monkeypatched at the module
level, same pattern as `test_master.py`; these tests never touch the
network or load a real model.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.critic import identity as critic_identity
from contracts import FeatureCheck, FeatureChecklistResult, LookCard
from ml.identity.faces import DetectedFaceResult


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


def _checklist(*, score: float = 1.0, mandatory_passed: bool = True) -> FeatureChecklistResult:
    return FeatureChecklistResult(
        checks=[FeatureCheck(feature="hair", expected="x", present=True, mandatory=True)],
        score=score,
        mandatory_passed=mandatory_passed,
        artifacts_clean=True,
        image_sha256="x",
        model="mock:mock",
    )


def _unit(vector: list[float]) -> np.ndarray:
    array = np.array(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


async def _async_result(value: object) -> object:
    return value


async def test_score_identity_is_checklist_dominant(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: None)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: None)
    monkeypatch.setattr(
        critic_identity, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=1.0))
    )

    score, face_box = await critic_identity.score_identity(b"fake-jpeg", _look_card())

    # no face detected -> no ArcFace term -> the 0.5 neutral default fills in,
    # but checklist (weight 0.7) still dominates the composite.
    expected = critic_identity.W_CHECKLIST * 1.0 + critic_identity.W_ARCFACE * 0.5
    assert score == pytest.approx(expected)
    assert face_box is None


async def test_score_identity_includes_weak_arcface_agreement_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=_unit([1.0, 0.0]), face_count=1
    )
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: (1, 2, 3, 4))
    monkeypatch.setattr(
        critic_identity, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=0.8))
    )

    reference_embedding = _unit([1.0, 0.0])  # identical -> cosine 1.0 -> agreement 1.0
    score, face_box = await critic_identity.score_identity(
        b"fake-jpeg", _look_card(), reference_embedding=reference_embedding
    )

    expected = critic_identity.W_CHECKLIST * 0.8 + critic_identity.W_ARCFACE * 1.0
    assert score == pytest.approx(expected)
    assert face_box == (1, 2, 3, 4)


async def test_score_identity_discounts_a_small_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A small detected face (the character occupying only a couple
    percent of the frame) must have its checklist score pulled toward
    the neutral 0.5 midpoint rather than trusted outright — the
    checklist can't verify features it can't actually see. A 100x100
    image with a (0,0,20,10) face box is 2% of the frame, between
    `FACE_AREA_UNCERTAIN_FLOOR` (1.5%) and `FACE_AREA_CONFIDENT_THRESHOLD`
    (5%)."""
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=_unit([1.0, 0.0]), face_count=1
    )
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((100, 100, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: (0, 0, 20, 10))
    monkeypatch.setattr(
        critic_identity, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=1.0))
    )

    reference_embedding = _unit([1.0, 0.0])  # identical -> arcface agreement 1.0
    score, face_box = await critic_identity.score_identity(
        b"fake-jpeg", _look_card(), reference_embedding=reference_embedding
    )

    raw_score = critic_identity.W_CHECKLIST * 1.0 + critic_identity.W_ARCFACE * 1.0
    confidence = (0.02 - critic_identity.FACE_AREA_UNCERTAIN_FLOOR) / (
        critic_identity.FACE_AREA_CONFIDENT_THRESHOLD - critic_identity.FACE_AREA_UNCERTAIN_FLOOR
    )
    expected = confidence * raw_score + (1 - confidence) * critic_identity.UNCERTAIN_SCORE
    assert score == pytest.approx(expected)
    assert score < raw_score  # the discount must actually reduce a falsely-perfect score
    assert face_box == (0, 0, 20, 10)


async def test_score_identity_treats_a_tiny_face_as_fully_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """At or below `FACE_AREA_UNCERTAIN_FLOOR`, the checklist's verdict
    must be fully replaced by the neutral midpoint, regardless of how
    confident the (unverifiable) checklist score was."""
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=_unit([1.0, 0.0]), face_count=1
    )
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((100, 100, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: (0, 0, 10, 10))
    monkeypatch.setattr(
        critic_identity, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=1.0))
    )

    score, _face_box = await critic_identity.score_identity(
        b"fake-jpeg", _look_card(), reference_embedding=_unit([1.0, 0.0])
    )

    assert score == pytest.approx(critic_identity.UNCERTAIN_SCORE)


async def test_score_identity_does_not_discount_a_large_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A face at/above `FACE_AREA_CONFIDENT_THRESHOLD` (close/medium
    framing, the common case) must score exactly as before this
    discount existed — no regression for the well-covered majority."""
    fake_detected = DetectedFaceResult(
        crop_bgr=np.zeros((4, 4, 3), np.uint8), embedding=_unit([1.0, 0.0]), face_count=1
    )
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((100, 100, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: fake_detected)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: (0, 0, 30, 30))
    monkeypatch.setattr(
        critic_identity, "score_checklist", lambda *_a, **_k: _async_result(_checklist(score=0.8))
    )

    score, _face_box = await critic_identity.score_identity(
        b"fake-jpeg", _look_card(), reference_embedding=_unit([1.0, 0.0])
    )

    expected = critic_identity.W_CHECKLIST * 0.8 + critic_identity.W_ARCFACE * 1.0
    assert score == pytest.approx(expected)


async def test_score_identity_still_scores_with_no_detected_face(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the same real bug class fixed in
    `master.py`'s `rank_candidates`: a "no face" result from InsightFace
    must not block the checklist from running on the full image."""
    checklist_calls = []
    monkeypatch.setattr(
        critic_identity, "decode_jpeg_bgr", lambda data: np.zeros((4, 4, 3), np.uint8)
    )
    monkeypatch.setattr(critic_identity, "detect_and_embed", lambda image_bgr: None)
    monkeypatch.setattr(critic_identity, "detect_face_box", lambda image_bgr: None)

    async def fake_score_checklist(image: bytes, look_card: LookCard) -> FeatureChecklistResult:
        checklist_calls.append(image)
        return _checklist(score=0.9)

    monkeypatch.setattr(critic_identity, "score_checklist", fake_score_checklist)

    score, face_box = await critic_identity.score_identity(b"fake-jpeg", _look_card())

    assert len(checklist_calls) == 1  # the checklist still ran
    assert score > 0
    assert face_box is None
