"""critic/reward_head.py — the hand-set linear combo reward head. Pure
function over `Candidate.scores`, no model, no network.
"""

from __future__ import annotations

import pytest

from app.critic.reward_head import WEIGHTS, score
from contracts import Candidate


def _candidate(**scores: float) -> Candidate:
    return Candidate(id="c1", panel_id=1, url="https://example.test/c1.png", seed=1, scores=scores)


def test_weights_sum_to_one() -> None:
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_score_is_the_weighted_sum() -> None:
    candidate = _candidate(identity=0.8, style=0.6, alignment=0.9, detail=0.4)
    expected = 0.35 * 0.8 + 0.25 * 0.6 + 0.25 * 0.9 + 0.15 * 0.4
    assert score(candidate) == pytest.approx(expected)


def test_score_is_deterministic() -> None:
    candidate = _candidate(identity=0.8, style=0.6, alignment=0.9, detail=0.4)
    assert score(candidate) == score(candidate)


def test_missing_signal_counts_as_zero_not_a_free_pass() -> None:
    candidate = _candidate(identity=1.0, style=1.0)  # alignment/detail missing entirely
    expected = 0.35 * 1.0 + 0.25 * 1.0 + 0.25 * 0.0 + 0.15 * 0.0
    assert score(candidate) == pytest.approx(expected)


def test_out_of_range_signal_is_clipped_not_allowed_to_dominate() -> None:
    """A signal returning > 1.0 (a bug or an adversarial input) must not
    blow past its own weight's max share."""
    hacked = _candidate(identity=50.0, style=0.0, alignment=0.0, detail=0.0)
    normal_max = _candidate(identity=1.0, style=0.0, alignment=0.0, detail=0.0)
    assert score(hacked) == score(normal_max)


def test_negative_signal_is_clipped_to_zero() -> None:
    candidate = _candidate(identity=-5.0, style=0.5, alignment=0.5, detail=0.5)
    expected = 0.35 * 0.0 + 0.25 * 0.5 + 0.25 * 0.5 + 0.15 * 0.5
    assert score(candidate) == pytest.approx(expected)


def test_the_vision_judge_counts_for_half_when_it_rated_the_candidate() -> None:
    """The reward is an equal blend of the four hand-set signals and the
    vision judge's ratings when those ratings exist."""
    signals = {"identity": 0.8, "style": 0.6, "alignment": 0.9, "detail": 0.4}
    base = score(_candidate(**signals))
    judged = score(
        _candidate(
            **signals,
            judge_scene=1.0,
            judge_face=1.0,
            judge_body=1.0,
            judge_outfit=1.0,
            judge_overall=1.0,
        )
    )
    assert judged == pytest.approx(0.5 * base + 0.5 * 1.0)
    worse = score(_candidate(**signals, judge_scene=0.0, judge_overall=0.0))
    assert worse < base < judged  # the judge can lift a candidate and can sink it


def test_a_candidate_without_judge_ratings_is_scored_by_the_four_signals_alone() -> None:
    signals = {"identity": 0.8, "style": 0.6, "alignment": 0.9, "detail": 0.4}
    assert score(_candidate(**signals)) == pytest.approx(
        0.35 * 0.8 + 0.25 * 0.6 + 0.25 * 0.9 + 0.15 * 0.4
    )
