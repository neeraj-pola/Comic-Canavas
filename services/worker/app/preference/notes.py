"""Verbal taste notes.

The numeric knobs can only say "warmer" or "wider". A few weeks of taps and caption edits
also say things they can't ("dry captions, not puns", "calm faces at the gym"). Once there
are enough taps, a small LLM call turns what the app has measured into at most five short
notes, stored on the person's preference row. The script agent reads them as extra context
(`nodes/script.py`), so they steer framing, expression and caption tone.

Cheap on purpose: one call per person per week, and only after `NOTES_MIN_TAPS` taps spread
over at least `NOTES_MIN_DAYS` days — before that there is nothing real to summarise, and an
LLM asked to summarise nothing invents something.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import numpy as np
import psycopg
from psycopg.types.json import Json
from pydantic import BaseModel, Field, StringConstraints

from app.llm.base import Message
from app.llm.routing import resolve
from app.preference import knobs
from app.preference.axes import AXES, DIRECTION_LABELS, MAX_LEVEL

NOTES_MIN_TAPS = 30
NOTES_MIN_DAYS = 10
NOTES_REFRESH_DAYS = 7
MAX_NOTES = 5
MAX_CAPTION_EDITS = 20

Note = Annotated[str, StringConstraints(min_length=3, max_length=120)]


class TasteNotes(BaseModel):
    """What the model returns: a handful of short, plain-language notes about this person."""

    notes: list[Note] = Field(default_factory=list, max_length=MAX_NOTES)


SYSTEM_PROMPT = """You write a short "taste card" about one person who reads a daily comic \
diary, for the comic's scriptwriter.

You are given measured facts about which images they pick and how they edit captions. Write up \
to five notes, each one short plain sentence (under 100 characters) the scriptwriter can act on \
— about camera framing, facial expression, colour mood, or caption voice.

Rules:
- Only write what the facts support. A knob marked "not sure" gets no note. If nothing is \
supported, return an empty list.
- Never invent events, people or places. Never mention numbers, percentages, taps or models.
- Caption notes need at least three edits pointing the same way (shorter, drier, no puns, ...).
- Write about the person, e.g. "Prefers wide shots with the character small in the scene." """


def is_due(
    *,
    n_taps: int,
    first_tap_at: datetime | None,
    notes_updated_at: datetime | None,
    now: datetime,
) -> bool:
    """Enough taps, over enough days, and not refreshed in the last week."""
    if n_taps < NOTES_MIN_TAPS or first_tap_at is None:
        return False
    if now - first_tap_at < timedelta(days=NOTES_MIN_DAYS):
        return False
    return notes_updated_at is None or now - notes_updated_at >= timedelta(days=NOTES_REFRESH_DAYS)


def _knob_facts(state: dict[str, Any]) -> list[str]:
    mu = np.array(state.get("mu") or [])
    cov = np.array(state.get("cov") or [])
    if mu.size != 2 * knobs.K or cov.shape != (2 * knobs.K, 2 * knobs.K):
        return []
    seen = np.zeros((knobs.K, len(knobs.LEVELS)), dtype=bool)
    for axis, name in enumerate(AXES):
        for level in (state.get("seen") or {}).get(name, []):
            seen[axis, int(level) + MAX_LEVEL] = True
    facts = []
    for axis, name in enumerate(AXES):
        best = knobs.best_level(mu, cov, axis, seen)
        sure = knobs.p_better(mu, cov, axis, best, 0) if best != 0 else 0.0
        negative, positive = DIRECTION_LABELS[name][1], DIRECTION_LABELS[name][0]
        if best == 0 or sure < 0.8:
            facts.append(f"{name}: not sure yet")
            continue
        word = positive if best > 0 else negative
        strength = "clearly" if abs(best) >= 2 else "somewhat"
        facts.append(f"{name}: prefers {word} ({strength}), fairly sure")
    return facts


def build_facts(conn: psycopg.Connection[Any], user_id: str, knob_state: dict[str, Any]) -> str:
    lines = ["Image picks (style knobs):", *[f"- {f}" for f in _knob_facts(knob_state)]]
    edits = conn.execute(
        """
        SELECT cp.original, cp.edited FROM caption_pairs cp
        JOIN days d ON d.job_id = cp.day_id
        WHERE d.user_id = %s ORDER BY cp.created_at DESC LIMIT %s
        """,
        (user_id, MAX_CAPTION_EDITS),
    ).fetchall()
    lines.append("")
    if edits:
        lines.append("Caption edits (original -> what they changed it to), newest first:")
        lines.extend(f'- "{e["original"]}" -> "{e["edited"]}"' for e in edits)
    else:
        lines.append("Caption edits: none yet.")
    return "\n".join(lines)


async def maybe_refresh_notes(
    conn: psycopg.Connection[Any],
    user_id: str,
    *,
    now: datetime | None = None,
    force: bool = False,
) -> bool:
    """Regenerate the notes if they are due. Returns whether a new set was written.
    Never raises into the caller's tap handling — the caller wraps it, but a missing row or
    an LLM failure just means the previous notes stay."""
    now = now or datetime.now(UTC)
    row = conn.execute(
        "SELECT n_taps, knobs, notes FROM preference_models WHERE user_id = %s", (user_id,)
    ).fetchone()
    if row is None:
        return False
    first = conn.execute(
        "SELECT min(tapped_at) AS first FROM preference_snapshots WHERE user_id = %s", (user_id,)
    ).fetchone()
    previous = row["notes"] or {}
    updated = previous.get("updated_at")
    if not force and not is_due(
        n_taps=int(row["n_taps"]),
        first_tap_at=first["first"] if first else None,
        notes_updated_at=datetime.fromisoformat(updated) if updated else None,
        now=now,
    ):
        return False

    facts = build_facts(conn, user_id, row["knobs"] or {})
    provider, model = resolve("prompts")
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": facts},
    ]
    result, _meta = await provider.structured(messages, TasteNotes, model=model, temperature=0.2)
    conn.execute(
        "UPDATE preference_models SET notes = %s WHERE user_id = %s",
        (
            Json(
                {
                    "items": result.notes,
                    "updated_at": now.isoformat(),
                    "n_taps": int(row["n_taps"]),
                }
            ),
            user_id,
        ),
    )
    return True
