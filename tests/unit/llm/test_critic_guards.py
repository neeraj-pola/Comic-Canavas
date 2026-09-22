"""critic/guards.py — anti-hacking guards (framing-diversity bonus, OCR
text penalty). `_get_engine` is monkeypatched to a fake RapidOCR engine;
these tests never load the real ONNX models.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.critic import guards


class _FakeResult:
    def __init__(self, boxes: object) -> None:
        self.boxes = boxes


class _FakeEngine:
    def __init__(self, boxes: object) -> None:
        self._boxes = boxes
        self.calls: list[np.ndarray] = []

    def __call__(self, image: np.ndarray) -> _FakeResult:
        self.calls.append(image)
        return _FakeResult(self._boxes)


def _image() -> np.ndarray:
    return np.zeros((10, 10, 3), dtype=np.uint8)


def test_has_detected_text_is_false_when_boxes_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """RapidOCR's docs describe an empty array for "no text", but its
    real 2.x behavior is `result.boxes is None`."""
    monkeypatch.setattr(guards, "_get_engine", lambda: _FakeEngine(boxes=None))

    assert guards.has_detected_text(_image()) is False


def test_has_detected_text_is_false_for_an_empty_boxes_array(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guards, "_get_engine", lambda: _FakeEngine(boxes=np.empty((0, 4, 2))))

    assert guards.has_detected_text(_image()) is False


def test_has_detected_text_is_true_when_boxes_are_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(guards, "_get_engine", lambda: _FakeEngine(boxes=np.zeros((2, 4, 2))))

    assert guards.has_detected_text(_image()) is True


def test_framing_diversity_bonus_favors_wide_over_close() -> None:
    assert guards.framing_diversity_bonus("wide") > guards.framing_diversity_bonus("medium")
    assert guards.framing_diversity_bonus("medium") > guards.framing_diversity_bonus("close")
    assert guards.framing_diversity_bonus("close") == 0.0


def test_framing_diversity_bonus_is_zero_with_no_script_context() -> None:
    assert guards.framing_diversity_bonus(None) == 0.0


def test_text_penalty_is_zero_when_no_text_detected() -> None:
    assert guards.text_penalty(False) == 0.0


def test_text_penalty_is_positive_when_text_detected() -> None:
    assert guards.text_penalty(True) == guards.TEXT_PENALTY
    assert guards.TEXT_PENALTY > 0.0


def test_guarded_reward_adds_framing_bonus() -> None:
    base = 0.5
    assert guards.guarded_reward(base, framing="wide") == pytest.approx(
        base + guards.FRAMING_BONUS["wide"]
    )


def test_guarded_reward_subtracts_text_penalty() -> None:
    base = 0.5
    assert guards.guarded_reward(base, has_text=True) == pytest.approx(base - guards.TEXT_PENALTY)


def test_guarded_reward_never_goes_negative() -> None:
    assert guards.guarded_reward(0.01, has_text=True) == 0.0


def test_guarded_reward_defaults_match_the_unguarded_score() -> None:
    assert guards.guarded_reward(0.5) == pytest.approx(0.5)
