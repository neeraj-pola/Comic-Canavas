"""Writes a finished `DayState` into the relational tables the API reads (`days`, `beats`,
`panels`, `candidates`, `image_prompts`). Every insert is `ON CONFLICT ... DO UPDATE`, so
re-running a job never fails on a duplicate key."""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg.types.json import Json

from contracts import Candidate, DayState, ImagePrompt


def persist_day_state(
    conn: psycopg.Connection[Any],
    state: DayState,
    *,
    story_url: str | None,
    kind: str = "daily",
) -> None:
    """Upserts a day's row plus its beats, panels, prompts and candidates."""
    quiet_day = state.script.quiet_day if state.script else None
    conn.execute(
        """
        INSERT INTO days (
            job_id, user_id, date, source, text, audio_url, transcript,
            mood, quiet_day, strip_url, story_url, layout, cost_usd, errors, versions, kind
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (job_id) DO UPDATE SET
            source = EXCLUDED.source,
            text = EXCLUDED.text,
            audio_url = EXCLUDED.audio_url,
            transcript = EXCLUDED.transcript,
            mood = EXCLUDED.mood,
            quiet_day = EXCLUDED.quiet_day,
            strip_url = EXCLUDED.strip_url,
            story_url = EXCLUDED.story_url,
            layout = EXCLUDED.layout,
            cost_usd = EXCLUDED.cost_usd,
            errors = EXCLUDED.errors,
            versions = EXCLUDED.versions
        """,
        (
            state.job_id,
            state.user_id,
            state.date,
            state.source,
            state.text,
            state.audio_url,
            state.transcript,
            state.script.mood if state.script else None,
            quiet_day,
            state.strip_url,
            story_url,
            Json(state.layout) if state.layout is not None else None,
            state.cost_usd,
            state.errors,
            Json(state.versions),
            kind,
        ),
    )

    if state.beats is not None:
        for beat in state.beats.beats:
            conn.execute(
                """
                INSERT INTO beats (
                    day_id, beat_id, time, place, place_detail, event, emotion,
                    people, objects, importance, humor, quote
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (day_id, beat_id) DO UPDATE SET
                    time = EXCLUDED.time,
                    place = EXCLUDED.place,
                    place_detail = EXCLUDED.place_detail,
                    event = EXCLUDED.event,
                    emotion = EXCLUDED.emotion,
                    people = EXCLUDED.people,
                    objects = EXCLUDED.objects,
                    importance = EXCLUDED.importance,
                    humor = EXCLUDED.humor,
                    quote = EXCLUDED.quote
                """,
                (
                    state.job_id,
                    beat.id,
                    beat.time,
                    beat.place,
                    beat.place_detail,
                    beat.event,
                    beat.emotion,
                    beat.people,
                    beat.objects,
                    beat.importance,
                    beat.humor,
                    beat.quote,
                ),
            )

    if state.script is not None:
        for panel in state.script.panels:
            conn.execute(
                """
                INSERT INTO panels (
                    day_id, panel_id, beat_id, place, time_of_day, expression,
                    action, framing, caption_a, caption_b, bubble, cast_ids
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (day_id, panel_id) DO UPDATE SET
                    beat_id = EXCLUDED.beat_id,
                    place = EXCLUDED.place,
                    time_of_day = EXCLUDED.time_of_day,
                    expression = EXCLUDED.expression,
                    action = EXCLUDED.action,
                    framing = EXCLUDED.framing,
                    caption_a = EXCLUDED.caption_a,
                    caption_b = EXCLUDED.caption_b,
                    bubble = EXCLUDED.bubble,
                    cast_ids = EXCLUDED.cast_ids
                """,
                (
                    state.job_id,
                    panel.id,
                    panel.beat_id,
                    panel.place,
                    panel.time_of_day,
                    panel.expression,
                    panel.action,
                    panel.framing,
                    panel.caption_a,
                    panel.caption_b,
                    panel.bubble,
                    panel.cast,
                ),
            )

    prompt_by_panel = persist_image_prompts(conn, state.job_id, state.prompts)
    persist_candidates(conn, state.job_id, state.candidates, prompt_by_panel=prompt_by_panel)


def persist_image_prompts(
    conn: psycopg.Connection[Any], day_id: str, prompts: list[ImagePrompt]
) -> dict[int, str]:
    """Upserts the last prompt per panel (a retry's earlier attempts don't survive) and
    returns a `panel_id -> prompt_id` map for `persist_candidates` to link against."""
    last_by_panel: dict[int, ImagePrompt] = {}
    for prompt in prompts:
        last_by_panel[prompt.panel_id] = prompt  # later entries (retries) win

    prompt_ids: dict[int, str] = {}
    for panel_id, prompt in last_by_panel.items():
        prompt_id = f"{day_id}-p{panel_id}"
        conn.execute(
            """
            INSERT INTO image_prompts (
                id, day_id, panel_id, generator, character_clause, environment_clause,
                positive, negative, seed, guidance, steps
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                generator = EXCLUDED.generator,
                character_clause = EXCLUDED.character_clause,
                environment_clause = EXCLUDED.environment_clause,
                positive = EXCLUDED.positive,
                negative = EXCLUDED.negative,
                seed = EXCLUDED.seed,
                guidance = EXCLUDED.guidance,
                steps = EXCLUDED.steps
            """,
            (
                prompt_id,
                day_id,
                panel_id,
                prompt.generator,
                prompt.character_clause,
                prompt.environment_clause,
                prompt.positive,
                prompt.negative,
                prompt.seed,
                prompt.guidance,
                prompt.steps,
            ),
        )
        prompt_ids[panel_id] = prompt_id
    return prompt_ids


def persist_candidates(
    conn: psycopg.Connection[Any],
    day_id: str,
    candidates: list[Candidate],
    *,
    prompt_by_panel: dict[int, str] | None = None,
) -> None:
    """Upserts candidates. Separate from `persist_day_state` so a single panel's regenerated
    batch can be saved without rewriting the whole day."""
    prompt_by_panel = prompt_by_panel or {}
    for candidate in candidates:
        conn.execute(
            """
            INSERT INTO candidates (
                id, day_id, panel_id, url, seed, scores, face_box, chosen,
                rejected_reason, prompt_id, embedding
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                embedding = COALESCE(EXCLUDED.embedding, candidates.embedding),
                scores = EXCLUDED.scores,
                face_box = EXCLUDED.face_box,
                chosen = EXCLUDED.chosen,
                rejected_reason = EXCLUDED.rejected_reason,
                prompt_id = COALESCE(EXCLUDED.prompt_id, candidates.prompt_id)
            """,
            (
                candidate.id,
                day_id,
                candidate.panel_id,
                candidate.url,
                candidate.seed,
                Json(candidate.scores),
                list(candidate.face_box) if candidate.face_box is not None else None,
                candidate.chosen,
                candidate.rejected_reason,
                prompt_by_panel.get(candidate.panel_id),
                Json(candidate.embedding) if candidate.embedding is not None else None,
            ),
        )


def mark_job_status(
    conn: psycopg.Connection[Any], job_id: str, *, status: str, error: str | None = None
) -> None:
    """Updates a job's status and optional error."""
    conn.execute(
        "UPDATE jobs SET status = %s, error = %s, updated_at = now() WHERE id = %s",
        (status, error, job_id),
    )


def append_job_event(conn: psycopg.Connection[Any], job_id: str, event: dict[str, object]) -> None:
    """Appends one entry to a job's event log, read back by the progress SSE stream."""
    conn.execute(
        "UPDATE jobs SET events = events || %s::jsonb, updated_at = now() WHERE id = %s",
        (Json([event]), job_id),
    )
