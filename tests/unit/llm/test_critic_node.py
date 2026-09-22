"""nodes/critic.py — picks the best candidate per panel via the reward
head, retrying panels that fail the identity hard gate up to
`MAX_RETRIES` times before falling back to the best available candidate.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from app.critic.guards import FRAMING_BONUS, TEXT_PENALTY
from app.nodes.critic import (
    IDENTITY_THRESHOLD,
    MAX_RETRIES,
    choose_best_candidates,
    choose_best_for_panel,
)
from contracts import Candidate, DayState, Panel, Script


def _candidate(
    cand_id: str,
    panel_id: int,
    *,
    identity: float,
    reward_boost: float = 0.0,
    text_detected: bool = False,
) -> Candidate:
    scores = {
        "identity": identity,
        "style": 0.5 + reward_boost,
        "alignment": 0.5,
        "detail": 0.5,
    }
    if text_detected:
        scores["text_detected"] = 1.0
    return Candidate(
        id=cand_id,
        panel_id=panel_id,
        url=f"https://example.test/{cand_id}.png",
        seed=1,
        scores=scores,
    )


def _day_state(candidates: list[Candidate], *, script: Script | None = None) -> DayState:
    return DayState(
        job_id="job-1",
        user_id="u1",
        date=date(2026, 3, 1),
        source="text",
        candidates=candidates,
        script=script,
    )


def _panel(panel_id: int, framing: Literal["wide", "medium", "close"]) -> Panel:
    return Panel(
        id=panel_id,
        beat_id="b1",
        place="kitchen",
        time_of_day="morning",
        expression="content",
        action="pouring coffee",
        framing=framing,
        caption_a="Coffee time",
        caption_b="",
    )


async def test_a_passing_candidate_is_chosen_immediately_without_retry() -> None:
    candidates = [
        _candidate("c1", 1, identity=IDENTITY_THRESHOLD + 0.1),
        _candidate("c2", 1, identity=0.1),
    ]
    called = False

    async def regenerate(panel_id: int) -> list[Candidate]:
        nonlocal called
        called = True
        return []

    final, errors = await choose_best_for_panel(1, candidates, regenerate=regenerate)

    assert called is False
    assert errors == []
    assert [c.id for c in final] == ["c1", "c2"]
    chosen = [c for c in final if c.chosen]
    assert [c.id for c in chosen] == ["c1"]


async def test_the_higher_reward_candidate_wins_among_passers() -> None:
    candidates = [
        _candidate("low", 1, identity=IDENTITY_THRESHOLD + 0.1, reward_boost=0.0),
        _candidate("high", 1, identity=IDENTITY_THRESHOLD + 0.1, reward_boost=0.4),
    ]

    async def regenerate(panel_id: int) -> list[Candidate]:
        raise AssertionError("should not retry when a candidate already passes")

    final, errors = await choose_best_for_panel(1, candidates, regenerate=regenerate)

    assert errors == []
    chosen = [c.id for c in final if c.chosen]
    assert chosen == ["high"]


async def test_every_returned_candidate_gets_a_real_reward_score() -> None:
    """`reward` must be written back into `Candidate.scores` — the dict
    that's persisted to the DB and read by the frontend — not just used
    for internal ranking. Both the chosen and the rejected candidate
    must carry their own distinct reward score."""
    candidates = [
        _candidate("low", 1, identity=IDENTITY_THRESHOLD + 0.1, reward_boost=0.0),
        _candidate("high", 1, identity=IDENTITY_THRESHOLD + 0.1, reward_boost=0.4),
    ]

    async def regenerate(panel_id: int) -> list[Candidate]:
        raise AssertionError("should not retry when a candidate already passes")

    final, _errors = await choose_best_for_panel(1, candidates, regenerate=regenerate)

    by_id = {c.id: c for c in final}
    assert "reward" in by_id["low"].scores
    assert "reward" in by_id["high"].scores
    assert by_id["high"].scores["reward"] > by_id["low"].scores["reward"]


async def test_all_failing_retries_until_max_then_falls_back_with_error() -> None:
    attempts: list[int] = []

    async def regenerate(panel_id: int) -> list[Candidate]:
        attempts.append(panel_id)
        return [_candidate(f"retry{len(attempts)}", panel_id, identity=0.1)]

    initial = [_candidate("c1", 1, identity=0.2)]

    final, errors = await choose_best_for_panel(1, initial, regenerate=regenerate)

    assert len(attempts) == MAX_RETRIES
    assert len(errors) == 1
    assert "panel 1" in errors[0]
    assert str(IDENTITY_THRESHOLD) in errors[0]
    assert f"after {MAX_RETRIES} retries" in errors[0]
    chosen = [c for c in final if c.chosen]
    assert len(chosen) == 1


async def test_fallback_uses_the_final_retry_batch_not_the_original_candidates() -> None:
    """A retry discards the failed batch entirely — the winner (and every
    returned candidate) must come from the last regenerate() call, never
    from the original, now-stale candidates."""

    async def regenerate(panel_id: int) -> list[Candidate]:
        return [_candidate("final-batch", panel_id, identity=0.1)]

    initial = [_candidate("original", 1, identity=0.1)]

    final, errors = await choose_best_for_panel(1, initial, regenerate=regenerate)

    assert [c.id for c in final] == ["final-batch"]
    assert errors != []


async def test_a_late_pass_on_the_last_retry_stops_early_without_exhausting_error() -> None:
    calls = 0

    async def regenerate(panel_id: int) -> list[Candidate]:
        nonlocal calls
        calls += 1
        return [_candidate("saved", panel_id, identity=IDENTITY_THRESHOLD + 0.05)]

    initial = [_candidate("c1", 1, identity=0.1)]

    final, errors = await choose_best_for_panel(1, initial, regenerate=regenerate)

    assert calls == 1
    assert errors == []
    assert [c.id for c in final if c.chosen] == ["saved"]


async def test_choose_best_candidates_groups_by_panel_and_preserves_prior_errors() -> None:
    candidates = [
        _candidate("p1-a", 1, identity=IDENTITY_THRESHOLD + 0.1),
        _candidate("p1-b", 1, identity=0.1),
        _candidate("p2-a", 2, identity=IDENTITY_THRESHOLD + 0.1),
    ]
    state = _day_state(candidates)
    state = state.model_copy(update={"errors": ["pre-existing error from an earlier node"]})

    async def regenerate(panel_id: int) -> list[Candidate]:
        raise AssertionError("no candidate here should need a retry")

    result = await choose_best_candidates(state, regenerate=regenerate)

    assert result.errors == ["pre-existing error from an earlier node"]
    assert {c.panel_id for c in result.candidates} == {1, 2}
    chosen_ids = {c.id for c in result.candidates if c.chosen}
    assert chosen_ids == {"p1-a", "p2-a"}


async def test_choose_best_candidates_appends_fallback_errors_for_failing_panels() -> None:
    candidates = [
        _candidate("p1-a", 1, identity=IDENTITY_THRESHOLD + 0.1),
        _candidate("p2-a", 2, identity=0.1),
    ]
    state = _day_state(candidates)

    async def regenerate(panel_id: int) -> list[Candidate]:
        return [_candidate("p2-retry", panel_id, identity=0.1)]

    result = await choose_best_candidates(state, regenerate=regenerate)

    assert len(result.errors) == 1
    assert "panel 2" in result.errors[0]
    panel2_ids = {c.id for c in result.candidates if c.panel_id == 2}
    assert panel2_ids == {"p2-retry"}


async def test_wide_framing_bonus_can_beat_a_close_candidate_with_equal_base_reward() -> None:
    """Without the framing bonus these two candidates would tie exactly —
    the wide framing's bonus breaks the tie, offsetting identity's
    structural bias toward tight face crops."""
    wide = _candidate("wide-c", 1, identity=IDENTITY_THRESHOLD + 0.1)
    close = _candidate("close-c", 1, identity=IDENTITY_THRESHOLD + 0.1)
    script = Script(mood="ok", quiet_day=False, panels=[_panel(1, "wide"), _panel(2, "medium")])
    state = _day_state([wide, close], script=script)

    async def regenerate(panel_id: int) -> list[Candidate]:
        raise AssertionError("both candidates already pass, no retry needed")

    result = await choose_best_candidates(state, regenerate=regenerate)

    chosen = [c.id for c in result.candidates if c.chosen]
    assert chosen == ["wide-c"]


async def test_no_script_wired_means_no_framing_bonus_and_prior_behavior_holds() -> None:
    candidates = [
        _candidate("a", 1, identity=IDENTITY_THRESHOLD + 0.1),
        _candidate("b", 1, identity=IDENTITY_THRESHOLD + 0.1),
    ]
    final, errors = await choose_best_for_panel(
        1, candidates, regenerate=lambda panel_id: (_ for _ in ()).throw(AssertionError())
    )

    assert errors == []
    assert len([c for c in final if c.chosen]) == 1


async def test_detected_text_penalizes_a_candidate_below_a_clean_alternative() -> None:
    clean = _candidate("clean", 1, identity=IDENTITY_THRESHOLD + 0.1)
    texted = _candidate(
        "texted", 1, identity=IDENTITY_THRESHOLD + 0.1, reward_boost=0.1, text_detected=True
    )
    assert TEXT_PENALTY > 0.1  # the penalty must be large enough to flip this ranking

    final, _errors = await choose_best_for_panel(
        1, [clean, texted], regenerate=lambda panel_id: (_ for _ in ()).throw(AssertionError())
    )

    chosen = [c.id for c in final if c.chosen]
    assert chosen == ["clean"]


async def test_framing_bonus_values_are_wired_from_guards_module() -> None:
    """Confirms nodes/critic.py doesn't hardcode its own bonus values —
    changing critic/guards.py's FRAMING_BONUS should be the only place
    to tune this."""
    a = _candidate("a", 1, identity=IDENTITY_THRESHOLD + 0.1)
    state = _day_state(
        [a],
        script=Script(mood="ok", quiet_day=False, panels=[_panel(1, "wide"), _panel(2, "close")]),
    )

    async def regenerate(panel_id: int) -> list[Candidate]:
        raise AssertionError("no retry expected")

    result = await choose_best_candidates(state, regenerate=regenerate)

    assert FRAMING_BONUS["wide"] > 0.0
    assert result.candidates[0].chosen is True
