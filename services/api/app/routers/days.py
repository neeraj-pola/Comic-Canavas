"""`POST /days` enqueues the real daily job (`app.worker.run_daily_job`),
`GET /jobs/{id}/events` streams its progress over SSE, `GET /days/{date}`
reads back what `app.persist.persist_day_state` (the worker) wrote.

SSE is hand-rolled (`StreamingResponse` + `text/event-stream`, no new
dependency) rather than a client library: the worker and this process only
share Postgres (`jobs.events`, `jobs.status`) and Redis (arq), not an
in-process callback, so streaming here just means polling that one row at a
short interval and forwarding new entries.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import date as date_cls
from typing import Any

from arq import ArqRedis
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from psycopg.rows import DictRow

from app.db import get_conn
from app.deps import arq_pool_dep, current_user_id

router = APIRouter(tags=["days"])

# Only "pending"/"running" are genuinely in-flight and worth deduplicating
# against; "done" must fall through to a fresh enqueue so "Redo today's
# strip" actually starts a new run (`run_daily_job`'s own `is_redo` check
# then handles resetting the LangGraph checkpoint correctly).
_IN_FLIGHT_JOB_STATUSES = {"pending", "running"}


@router.post("/days", status_code=202)
async def create_day(
    body: dict[str, str | None],
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, str]:
    text = body.get("text")
    audio_url = body.get("audio_url")
    if not text and not audio_url:
        raise HTTPException(status_code=422, detail="one of text or audio_url is required")

    # The date is optional and defaults to today (lets a past day be
    # backfilled, e.g. a whole week for the weekly recap); the future is
    # rejected — a diary entry can't describe a day that hasn't happened.
    today = date_cls.today()
    raw_date = body.get("date")
    try:
        entry_date = date_cls.fromisoformat(raw_date) if raw_date else today
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD") from exc
    if entry_date > today:
        raise HTTPException(status_code=422, detail="date can't be in the future")
    job_date = entry_date.isoformat()
    job_id = f"{user_id}:{job_date}"

    # A separate SELECT then INSERT would be a TOCTOU race: two
    # near-simultaneous requests could both see "not in flight" and both
    # enqueue. This is atomic instead: Postgres only performs the conflict
    # update (and returns a row) when the WHERE condition holds, so a
    # concurrent loser's guard fails, RETURNING yields nothing, and that
    # request reports the winner's real in-flight status instead of
    # enqueueing a duplicate.
    with get_conn() as conn:
        won = conn.execute(
            """
            INSERT INTO jobs (id, user_id, kind, status, events)
            VALUES (%s, %s, 'daily', 'pending', '[]')
            ON CONFLICT (id) DO UPDATE SET status = 'pending', error = NULL, events = '[]'
            WHERE jobs.status <> ALL(%s)
            RETURNING status
            """,
            (job_id, user_id, list(_IN_FLIGHT_JOB_STATUSES)),
        ).fetchone()
        if won is None:
            existing = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
            assert existing is not None  # the conflict that lost the race guarantees a row exists
            return {"job_id": job_id, "status": existing["status"]}

    await arq_pool.enqueue_job(
        "run_daily_job", job_id, user_id, job_date, text=text, audio_url=audio_url
    )
    return {"job_id": job_id, "status": "pending"}


def _job_row(job_id: str, user_id: str) -> DictRow:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, status, error, events FROM jobs WHERE id = %s AND user_id = %s",
            (job_id, user_id),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="job not found")
    return row


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/jobs/{job_id}/events")
async def job_events(job_id: str, user_id: str = Depends(current_user_id)) -> StreamingResponse:
    await asyncio.to_thread(_job_row, job_id, user_id)  # 404 early, before opening the stream

    async def generator() -> AsyncIterator[str]:
        seen = 0
        while True:
            row = await asyncio.to_thread(_job_row, job_id, user_id)
            events = row["events"]
            for event in events[seen:]:
                yield _sse("node", event)
            seen = len(events)
            if row["status"] in ("done", "failed"):
                yield _sse("done", {"status": row["status"], "error": row["error"]})
                return
            await asyncio.sleep(0.3)

    return StreamingResponse(generator(), media_type="text/event-stream")


def _beat_out(row: DictRow) -> dict[str, Any]:
    return {
        "id": row["beat_id"],
        "time": row["time"],
        "place": row["place"],
        "place_detail": row["place_detail"],
        "event": row["event"],
        "emotion": row["emotion"],
        "people": row["people"],
        "objects": row["objects"],
        "importance": row["importance"],
        "humor": row["humor"],
        "quote": row["quote"],
    }


def _panel_out(row: DictRow) -> dict[str, Any]:
    return {
        "id": row["panel_id"],
        "beat_id": row["beat_id"],
        "place": row["place"],
        "time_of_day": row["time_of_day"],
        "expression": row["expression"],
        "action": row["action"],
        "framing": row["framing"],
        "caption_a": row["caption_a"],
        "caption_b": row["caption_b"],
        "bubble": row["bubble"],
        "cast": row["cast_ids"],
    }


def _candidate_out(row: DictRow) -> dict[str, Any]:
    return {
        "id": row["id"],
        "panel_id": row["panel_id"],
        "url": row["url"],
        "seed": row["seed"],
        "scores": row["scores"],
        "face_box": row["face_box"],
        "chosen": row["chosen"],
        "rejected_reason": row["rejected_reason"],
        "prompt_id": row["prompt_id"],
    }


def _prompt_out(row: DictRow) -> dict[str, Any]:
    return {
        "id": row["id"],
        "panel_id": row["panel_id"],
        "generator": row["generator"],
        "character_clause": row["character_clause"],
        "environment_clause": row["environment_clause"],
        "positive": row["positive"],
        "negative": row["negative"],
        "seed": row["seed"],
        "guidance": row["guidance"],
        "steps": row["steps"],
    }


@router.get("/days/{date}")
async def get_day(date: date_cls, user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    with get_conn() as conn:
        day = conn.execute(
            "SELECT * FROM days WHERE user_id = %s AND date = %s AND kind = 'daily'",
            (user_id, date),
        ).fetchone()
        if day is None:
            raise HTTPException(status_code=404, detail="no day for that date")
        beats = conn.execute(
            "SELECT * FROM beats WHERE day_id = %s ORDER BY beat_id", (day["job_id"],)
        ).fetchall()
        panels = conn.execute(
            "SELECT * FROM panels WHERE day_id = %s ORDER BY panel_id", (day["job_id"],)
        ).fetchall()
        candidates = conn.execute(
            "SELECT * FROM candidates WHERE day_id = %s ORDER BY panel_id, seed",
            (day["job_id"],),
        ).fetchall()
        prompts = conn.execute(
            "SELECT * FROM image_prompts WHERE day_id = %s ORDER BY panel_id",
            (day["job_id"],),
        ).fetchall()
        # What the person has already told the app about this day, so going back to it shows
        # their picks and ratings instead of an untouched picker.
        saved_picks = conn.execute(
            "SELECT DISTINCT ON (panel_id) panel_id, chosen FROM image_pairs "
            "WHERE day_id = %s AND source = 'ab' ORDER BY panel_id, created_at DESC, id DESC",
            (day["job_id"],),
        ).fetchall()
        saved_ratings = conn.execute(
            "SELECT DISTINCT ON (panel_id, url) url, rating FROM image_ratings "
            "WHERE day_id = %s ORDER BY panel_id, url, created_at DESC, id DESC",
            (day["job_id"],),
        ).fetchall()

    return {
        "date": day["date"].isoformat(),
        "source": day["source"],
        "text": day["text"],
        "transcript": day["transcript"],
        "mood": day["mood"],
        "quiet_day": day["quiet_day"],
        "strip_url": day["strip_url"],
        "story_url": day["story_url"],
        "layout": day["layout"],
        "cost_usd": day["cost_usd"],
        "versions": day["versions"],
        "beats": [_beat_out(b) for b in beats],
        "panels": [_panel_out(p) for p in panels],
        "candidates": [_candidate_out(c) for c in candidates],
        "prompts": [_prompt_out(p) for p in prompts],
        "picks": {str(r["panel_id"]): r["chosen"] for r in saved_picks},
        "ratings": {r["url"]: r["rating"] for r in saved_ratings},
    }
