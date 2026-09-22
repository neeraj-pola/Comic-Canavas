"""Choosing the week's highlights: valid picks kept, bad ones dropped, topped up."""

from __future__ import annotations

import pytest

from app.weekly_highlights import (
    MAX_PER_DAY,
    NoPanelsInWeekError,
    SourcePanel,
    WeekHighlights,
    build_messages,
    choose_highlights,
    validate,
)


def _panels(days: int = 5, per_day: int = 4) -> list[SourcePanel]:
    return [
        SourcePanel(
            date=f"2026-09-{7 + d:02d}",
            day_mood="content",
            panel_id=p,
            caption=f"Caption {d}-{p}",
            action="doing something",
            place="desk",
            time_of_day="morning",
            expression="content",
            framing="medium",
            url=f"http://x/{d}-{p}.png",
        )
        for d in range(days)
        for p in range(1, per_day + 1)
    ]


def _picked(*refs: int) -> WeekHighlights:
    return WeekHighlights.model_validate(
        {"mood": "a good week", "highlights": [{"ref": r, "caption": f"H{r}"} for r in refs]}
    )


def test_valid_picks_are_kept_in_calendar_order_with_their_captions() -> None:
    panels = _panels()
    out = validate(panels, _picked(9, 2, 14, 5, 18, 0), 6)
    assert [p.url for p, _ in out] == [panels[i].url for i in sorted([9, 2, 14, 5, 18, 0])]
    assert dict((p.url, c) for p, c in out)[panels[9].url] == "H9"


def test_invalid_and_repeated_refs_are_dropped_and_the_recap_is_topped_up() -> None:
    panels = _panels()
    out = validate(panels, _picked(2, 2, 99, 7), 6)
    assert len(out) == 6 and len({p.url for p, _ in out}) == 6
    assert panels[2].url in {p.url for p, _ in out} and panels[7].url in {p.url for p, _ in out}


def test_no_day_supplies_more_than_the_limit() -> None:
    panels = _panels()
    out = validate(panels, _picked(0, 1, 2, 3, 8, 12), 6)  # four picks from day one
    per_day: dict[str, int] = {}
    for p, _ in out:
        per_day[p.date] = per_day.get(p.date, 0) + 1
    assert max(per_day.values()) <= MAX_PER_DAY and len(out) == 6


def test_a_week_with_fewer_panels_than_six_keeps_them_all() -> None:
    panels = _panels(days=1, per_day=3)
    assert len(validate(panels, _picked(1), 3)) == 3


def test_the_prompt_lists_every_panel_with_its_number_and_asks_for_the_right_count() -> None:
    messages = build_messages(_panels(days=2, per_day=2), 3)
    body = str(messages[1]["content"])
    assert "Choose 3 highlights from these 4 panels" in body
    assert "[0] 2026-09-07" in body and "[3] 2026-09-08" in body and "Caption 1-2" in body


async def test_an_empty_week_raises_a_clear_error() -> None:
    with pytest.raises(NoPanelsInWeekError):
        await choose_highlights([])
