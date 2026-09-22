"""`GET /library?month=`, `GET /search?q=`.

`PostgresMemoryStore.search()` (in `services/worker`) is semantic, pgvector
cosine over `bge-small-en-v1.5` embeddings, but lives in a separate Python
namespace and needs `torch`/`transformers` — too heavy a dependency for
this one read endpoint. `/search` here is lexical (`ILIKE`) over the
relational `beats` table instead.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from app.db import get_conn
from app.deps import current_user_id

router = APIRouter(tags=["library"])


@router.get("/library")
async def library(
    month: str = Query(..., pattern=r"^\d{4}-\d{2}$"),
    mood: str | None = None,
    user_id: str = Depends(current_user_id),
) -> dict[str, Any]:
    try:
        year_str, month_str = month.split("-")
        start = date_cls(int(year_str), int(month_str), 1)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="invalid month, expected YYYY-MM") from exc
    end = date_cls(start.year + (start.month == 12), (start.month % 12) + 1, 1)

    with get_conn() as conn:
        days = conn.execute(
            """
            SELECT job_id, date, mood, quiet_day, strip_url
            FROM days
            WHERE user_id = %s AND kind = 'daily'
              AND date >= %s AND date < %s AND (%s::text IS NULL OR mood = %s)
            ORDER BY date
            """,
            (user_id, start, end, mood, mood),
        ).fetchall()
        job_ids = [d["job_id"] for d in days]
        thumbs_by_day: dict[str, list[int]] = {}
        if job_ids:
            for row in conn.execute(
                "SELECT day_id, value FROM thumbs WHERE day_id = ANY(%s)", (job_ids,)
            ).fetchall():
                thumbs_by_day.setdefault(row["day_id"], []).append(row["value"])

    return {
        "month": month,
        "days": [
            {
                "date": d["date"].isoformat(),
                "mood": d["mood"],
                "quiet_day": d["quiet_day"],
                "strip_url": d["strip_url"],
                "thumbs": thumbs_by_day.get(d["job_id"], []),
            }
            for d in days
        ],
    }


@router.get("/search")
async def search(
    q: str = Query(..., min_length=1), user_id: str = Depends(current_user_id)
) -> dict[str, Any]:
    like = f"%{q}%"
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.date, b.day_id, b.beat_id, b.event, b.place, b.quote,
                   COALESCE(
                       array_agg(c.url) FILTER (WHERE c.chosen AND c.url IS NOT NULL), '{}'
                   ) AS panel_urls
            FROM beats b
            JOIN days d ON d.job_id = b.day_id
            LEFT JOIN panels p ON p.day_id = b.day_id AND p.beat_id = b.beat_id
            LEFT JOIN candidates c ON c.day_id = p.day_id AND c.panel_id = p.panel_id
            WHERE d.user_id = %s AND d.kind = 'daily'
              AND (b.event ILIKE %s OR b.place ILIKE %s OR b.quote ILIKE %s)
            GROUP BY d.date, b.day_id, b.beat_id, b.event, b.place, b.quote
            ORDER BY d.date DESC
            """,
            (user_id, like, like, like),
        ).fetchall()

    return {
        "q": q,
        "hits": [
            {
                "date": r["date"].isoformat(),
                "beat_id": r["beat_id"],
                "event": r["event"],
                "place": r["place"],
                "quote": r["quote"],
                "panel_urls": r["panel_urls"],
            }
            for r in rows
        ],
    }
