"""Weekly recap graph — a separate graph from the daily `DayGraph`.

Takes the week's `BeatSheet`s, ranks every beat by `importance`, and builds a
6-panel `Script` from the top ones. For each panel, reuses that beat's
already-chosen daily panel if one exists (`MemoryStore.get_panel_urls`);
otherwise generates a fresh one through the same per-panel machinery the
daily graph's critic retry uses (`write_single_prompt` + `generate_one_panel`
+ `score_one_candidate` + `choose_best_for_panel`).

Builds one synthetic `DayState` as a carrier object so the per-panel helpers
(which only read `state.script`/`state.job_id`/`state.user_id`) work unchanged
for a non-daily caller. Composes via `compose_strip` directly (not
`compose_day`, which is hardcoded to the daily storage prefix and a full
`DayState`'s candidate list) into a 3x2 grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
from storage import Storage

from app.compose import compose_strip
from app.images.base import IdentityRef
from app.memory import MemoryStore, get_memory_store
from app.nodes.critic import choose_best_for_panel
from app.nodes.generate import generate_one_panel
from app.nodes.prompts import write_single_prompt
from app.nodes.score import score_one_candidate
from app.nodes.script import DEFAULT_HUMOR_LEVEL, write_script
from contracts import Beat, BeatSheet, Candidate, DayState, LookCard, Panel, Script

TOP_N_BEATS = 6
CANDIDATES_PER_MISSING_PANEL = 3


class NoBeatsInWeekError(ValueError):
    def __init__(self) -> None:
        super().__init__("build_weekly_recap: no beats across the given week's BeatSheets")


@dataclass
class WeeklyRecap:
    script: Script
    # The top beats under the synthetic ids ("w0"..) the script's panels refer
    # to. Save these, not the original per-day beats — `panels.beat_id` is a
    # foreign key into this renamed set.
    beats: list[Beat]
    candidates: list[Candidate]
    strip_url: str
    story_url: str
    layout: dict[str, Any]
    new_generations: int
    errors: list[str]


def select_top_beats(
    beat_sheets: list[BeatSheet], *, n: int = TOP_N_BEATS
) -> list[tuple[Beat, date]]:
    """Flattens every beat across the week, paired with the date of the
    `BeatSheet` it came from, and returns the top `n` by `importance`."""
    all_beats = [(beat, sheet.date) for sheet in beat_sheets for beat in sheet.beats]
    if not all_beats:
        raise NoBeatsInWeekError
    all_beats.sort(key=lambda pair: pair[0].importance, reverse=True)
    return all_beats[:n]


def _synthesize_week_beat_sheet(
    beat_sheets: list[BeatSheet], top_beats: list[Beat], *, week_end: date
) -> BeatSheet:
    mood_arc = [sheet.mood_arc[0] for sheet in beat_sheets if sheet.mood_arc][:6]
    people = sorted({p for sheet in beat_sheets for p in sheet.people_mentioned})
    flags = sorted({f for sheet in beat_sheets for f in sheet.flags})
    return BeatSheet(
        date=week_end,
        mood_arc=mood_arc or ["a look back at the week"],
        beats=top_beats,
        people_mentioned=people,
        flags=flags,
    )


async def _generate_missing_panel(
    panel: Panel,
    *,
    carrier: DayState,
    storage: Storage,
    identity: IdentityRef,
    look_card: LookCard,
    reference_embedding: np.ndarray | None,
    reference_bank: np.ndarray | None,
) -> tuple[Candidate, list[str]]:
    """Generates and scores `CANDIDATES_PER_MISSING_PANEL` candidates for one
    panel with no reusable daily image, then picks the best via the same
    retry-then-fallback logic the daily critic uses."""

    async def score_batch(fresh: list[Candidate]) -> list[Candidate]:
        return [
            await score_one_candidate(
                c,
                panel,
                storage=storage,
                look_card=look_card,
                reference_embedding=reference_embedding,
                reference_bank=reference_bank,
            )
            for c in fresh
        ]

    async def regenerate(panel_id: int) -> list[Candidate]:
        prompt, _usd = await write_single_prompt(carrier, panel_id, look_card=look_card)
        fresh = await generate_one_panel(
            prompt, storage=storage, identity=identity, n=CANDIDATES_PER_MISSING_PANEL
        )
        return await score_batch(fresh)

    initial = await regenerate(panel.id)
    final_candidates, errors = await choose_best_for_panel(
        panel.id, initial, regenerate=regenerate, framing=panel.framing
    )
    winner = next(c for c in final_candidates if c.chosen)
    return winner, errors


async def build_weekly_recap(
    user_id: str,
    beat_sheets: list[BeatSheet],
    *,
    storage: Storage,
    identity: IdentityRef,
    look_card: LookCard,
    iso_week: str,
    memory: MemoryStore | None = None,
    reference_embedding: np.ndarray | None = None,
    reference_bank: np.ndarray | None = None,
    humor_level: int = DEFAULT_HUMOR_LEVEL,
    taste_notes: list[str] | None = None,
) -> WeeklyRecap:
    if not beat_sheets:
        raise NoBeatsInWeekError

    memory_store = memory or get_memory_store()
    week_end = max(sheet.date for sheet in beat_sheets)
    top_pairs = select_top_beats(beat_sheets)
    # `Beat.id` is only unique within one day's `BeatSheet` (the extractor
    # restarts "b1", "b2" every day), so two of the week's top beats can share
    # an id. Every top beat is renamed to a synthetic, guaranteed-unique id
    # before it's handed to the script LLM; `original_beat_id_by_new_id` maps
    # back to the real per-day id `MemoryStore.get_panel_urls` needs.
    original_beat_id_by_new_id: dict[str, str] = {}
    beat_dates: dict[str, date] = {}
    top_beats: list[Beat] = []
    for i, (beat, day) in enumerate(top_pairs):
        new_id = f"w{i}"
        original_beat_id_by_new_id[new_id] = beat.id
        beat_dates[new_id] = day
        top_beats.append(beat.model_copy(update={"id": new_id}))

    week_sheet = _synthesize_week_beat_sheet(beat_sheets, top_beats, week_end=week_end)
    carrier = DayState(
        job_id=f"weekly-{iso_week}", user_id=user_id, date=week_end, source="text", beats=week_sheet
    )
    carrier = await write_script(
        carrier,
        memory=memory_store,
        humor_level=humor_level,
        max_panels=TOP_N_BEATS,
        taste_notes=taste_notes,
    )
    script = carrier.script
    assert script is not None  # write_script always sets it or raises

    candidates: list[Candidate] = []
    errors: list[str] = []
    new_generations = 0

    for panel in script.panels:
        beat_day = beat_dates.get(panel.beat_id, week_end)
        original_beat_id = original_beat_id_by_new_id.get(panel.beat_id, panel.beat_id)
        reused_urls = memory_store.get_panel_urls(user_id, original_beat_id, beat_day)
        if reused_urls:
            candidates.append(
                Candidate(
                    id=f"reused-{panel.id}-{panel.beat_id}",
                    panel_id=panel.id,
                    url=reused_urls[0],
                    seed=0,
                    chosen=True,
                )
            )
            continue

        new_generations += 1
        winner, panel_errors = await _generate_missing_panel(
            panel,
            carrier=carrier,
            storage=storage,
            identity=identity,
            look_card=look_card,
            reference_embedding=reference_embedding,
            reference_bank=reference_bank,
        )
        candidates.append(winner)
        errors.extend(panel_errors)

    images_by_panel = {c.panel_id: storage.get_object_by_url(c.url) for c in candidates}
    face_boxes = {c.panel_id: c.face_box for c in candidates}
    year_text, week_text = iso_week.split("-W")
    monday = date.fromisocalendar(int(year_text), int(week_text), 1)
    sunday = monday + timedelta(days=6)
    range_text = (
        f"{monday:%B %-d} \u2013 {sunday:%-d, %Y}"
        if monday.month == sunday.month
        else f"{monday:%B %-d} \u2013 {sunday:%B %-d, %Y}"
    )
    composed = compose_strip(
        images_by_panel,
        script,
        day=week_end,
        face_boxes=face_boxes,
        header_title=f"The week of {range_text}",
    )

    prefix = f"weekly/{user_id}/{iso_week}"
    strip_url = storage.put_object(
        f"{prefix}/strip.png", composed.strip_png, content_type="image/png"
    )
    story_url = storage.put_object(
        f"{prefix}/story.png", composed.story_png, content_type="image/png"
    )
    storage.put_object(
        f"{prefix}/layout.json",
        json.dumps(composed.layout, indent=2).encode(),
        content_type="application/json",
    )

    return WeeklyRecap(
        script=script,
        beats=top_beats,
        candidates=candidates,
        strip_url=strip_url,
        story_url=story_url,
        layout=composed.layout,
        new_generations=new_generations,
        errors=errors,
    )
