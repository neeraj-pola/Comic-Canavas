"""ml/identity/checklist.py — the VLM feature-checklist scorer. Uses the
mock LLM provider routed to the `judge` role, same pattern as
`test_lookcard.py`'s `lookcard` role.
"""

from __future__ import annotations

import json

import pytest

from app.llm import routing
from app.llm.mock_provider import MockProvider
from contracts import LookCard
from ml.identity.checklist import clear_cache, features_for, score_checklist


def _look_card(**overrides: object) -> LookCard:
    fields: dict[str, object] = {
        "hair": "short black wavy hair",
        "glasses": "black rectangular glasses",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "not enough information",
        "distinguishing": "a small mole above the left eyebrow",
        "gender_term": "man",
        "mandatory": ["glasses", "hair", "distinguishing"],
    }
    fields.update(overrides)
    return LookCard(**fields)


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("judge")
    assert isinstance(provider, MockProvider)
    return provider


def _answers_json(**overrides: object) -> str:
    payload = {
        "answers": [
            {"feature": "hair", "present": True, "note": "matches"},
            {"feature": "glasses", "present": True, "note": "matches"},
            {"feature": "skin_tone", "present": True, "note": "matches"},
            {"feature": "face_shape", "present": True, "note": "matches"},
            {"feature": "distinguishing", "present": True, "note": "matches"},
        ],
        "artifacts_clean": True,
        "artifact_note": "",
    }
    payload.update(overrides)
    return json.dumps(payload)


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_JUDGE", "mock:mock-judge")
    clear_cache()


def test_features_for_skips_unanswerable_lines() -> None:
    features = features_for(_look_card())
    keys = {f.key for f in features}
    assert "signature_outfit" not in keys  # "not enough information"
    assert "hair" in keys and "glasses" in keys


def test_features_for_marks_exactly_the_mandatory_fields() -> None:
    features = features_for(_look_card(mandatory=["glasses"]))
    mandatory_keys = {f.key for f in features if f.mandatory}
    assert mandatory_keys == {"glasses"}


async def test_score_checklist_computes_passed_over_total() -> None:
    provider = _resolved_mock_provider()
    # 4 present, 1 absent, out of 5 askable features (skin_tone/face_shape/
    # hair/glasses/distinguishing — signature_outfit is unanswerable)
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(
            answers=[
                {"feature": "hair", "present": True},
                {"feature": "glasses", "present": True},
                {"feature": "skin_tone", "present": True},
                {"feature": "face_shape", "present": True},
                {"feature": "distinguishing", "present": False},
            ]
        ),
    )
    result = await score_checklist(b"fake-jpeg", _look_card())
    assert result.score == pytest.approx(4 / 5)


async def test_mandatory_passed_is_false_when_a_mandatory_feature_fails() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(answers=[{"feature": "glasses", "present": False}]),
    )
    result = await score_checklist(b"fake-jpeg", _look_card(mandatory=["glasses"]))
    assert result.mandatory_passed is False


async def test_missing_answer_reconciles_to_present_false() -> None:
    provider = _resolved_mock_provider()
    # Answers only 4 of the 5 askable features — "distinguishing" missing.
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(
            answers=[
                {"feature": "hair", "present": True},
                {"feature": "glasses", "present": True},
                {"feature": "skin_tone", "present": True},
                {"feature": "face_shape", "present": True},
            ]
        ),
    )
    result = await score_checklist(b"fake-jpeg", _look_card())
    missing = next(c for c in result.checks if c.feature == "distinguishing")
    assert missing.present is False
    assert missing.note == "no answer"


async def test_unknown_extra_answer_is_dropped() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(
            answers=[
                {"feature": "hair", "present": True},
                {"feature": "glasses", "present": True},
                {"feature": "skin_tone", "present": True},
                {"feature": "face_shape", "present": True},
                {"feature": "distinguishing", "present": True},
                {"feature": "extra_thing_not_asked_for", "present": True},
            ]
        ),
    )
    result = await score_checklist(b"fake-jpeg", _look_card())
    assert {c.feature for c in result.checks} == {
        "hair",
        "glasses",
        "skin_tone",
        "face_shape",
        "distinguishing",
    }


async def test_matches_a_feature_echoed_with_its_description_attached() -> None:
    """A provider may echo `feature` back with its description attached
    ("hair: short black wavy hair" instead of just "hair") rather than
    verbatim. Reconciliation must still match this shape, as defense in
    depth against a provider doing this regardless
    of prompt wording."""
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(
            answers=[
                {"feature": "hair: short black wavy hair", "present": True},
                {"feature": "glasses: black rectangular glasses", "present": True},
                {"feature": "skin_tone: medium", "present": True},
                {"feature": "face_shape: oval", "present": True},
                {"feature": "distinguishing: a small mole above the left eyebrow", "present": True},
            ]
        ),
    )
    result = await score_checklist(b"fake-jpeg", _look_card())
    assert result.score == 1.0
    assert all(c.present for c in result.checks)
    assert result.mandatory_passed is True


async def test_artifacts_clean_surfaces_onto_the_result() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        _answers_json(artifacts_clean=False, artifact_note="second face visible"),
    )
    result = await score_checklist(b"fake-jpeg", _look_card())
    assert result.artifacts_clean is False
    assert result.artifact_note == "second face visible"


async def test_cache_hit_does_not_consume_a_second_canned_response() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("_VisionChecklistAnswers", [_answers_json(), "INVALID JSON"])

    look_card = _look_card()
    first = await score_checklist(b"fake-jpeg", look_card)
    second = await score_checklist(
        b"fake-jpeg", look_card
    )  # cache hit — must not touch "INVALID JSON"

    assert second == first


async def test_cache_miss_when_look_card_changes() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "_VisionChecklistAnswers",
        [
            _answers_json(answers=[{"feature": "glasses", "present": True}]),
            _answers_json(answers=[{"feature": "glasses", "present": False}]),
        ],
    )
    first = await score_checklist(b"fake-jpeg", _look_card(mandatory=["glasses"]))
    second = await score_checklist(
        b"fake-jpeg", _look_card(mandatory=["glasses"], glasses="round wire frames")
    )
    first_glasses = next(c for c in first.checks if c.feature == "glasses")
    second_glasses = next(c for c in second.checks if c.feature == "glasses")
    assert first_glasses.present is True
    assert second_glasses.present is False
