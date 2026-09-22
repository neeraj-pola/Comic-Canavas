"""`GET /weekly/{iso_week}` (recap/mood/stats/cast/training run) and
`GET /weekly/{iso_week}.pdf` (server-rendered via Playwright from
`templates/weekly.html`).

`POST /weekly/{iso_week}/generate` triggers the real worker job
(`app.worker.run_weekly_job`). The scheduler's own `weekly_recap` cron
entrypoint is what normally produces these automatically; this exists so a
recap can be generated on demand without waiting for Sunday.
"""

from __future__ import annotations

import re
from datetime import date as date_cls
from pathlib import Path
from typing import Any

from arq import ArqRedis
from fastapi import APIRouter, Depends, HTTPException, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape
from playwright.async_api import async_playwright

from app.db import get_conn
from app.deps import arq_pool_dep, current_user_id

router = APIRouter(prefix="/weekly", tags=["weekly"])

_ISO_WEEK_RE = re.compile(r"^\d{4}-W\d{2}$")
_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "templates"
_env = Environment(loader=FileSystemLoader(_TEMPLATES_DIR), autoescape=select_autoescape(["html"]))


def _validate_iso_week(iso_week: str) -> tuple[int, int]:
    if not _ISO_WEEK_RE.match(iso_week):
        raise HTTPException(status_code=422, detail="expected ISO week format YYYY-Www")
    year_str, week_str = iso_week.split("-W")
    return int(year_str), int(week_str)


def _week_job_id(user_id: str, iso_week: str) -> str:
    return f"{user_id}:weekly:{iso_week}"


@router.post("/{iso_week}/generate", status_code=202)
async def generate_weekly(
    iso_week: str,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, str]:
    _validate_iso_week(iso_week)
    job_id = _week_job_id(user_id, iso_week)
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO jobs (id, user_id, kind, status, events)
            VALUES (%s, %s, 'weekly', 'pending', '[]')
            ON CONFLICT (id) DO UPDATE SET status = 'pending', error = NULL, events = '[]'
            """,
            (job_id, user_id),
        )
    await arq_pool.enqueue_job("run_weekly_job", job_id, user_id, iso_week)
    return {"job_id": job_id, "status": "pending"}


def _weekly_context(user_id: str, iso_week: str) -> dict[str, Any]:
    year, week = _validate_iso_week(iso_week)
    monday = date_cls.fromisocalendar(year, week, 1)
    sunday = date_cls.fromisocalendar(year, week, 7)
    job_id = _week_job_id(user_id, iso_week)

    with get_conn() as conn:
        day = conn.execute(
            "SELECT * FROM days WHERE job_id = %s AND user_id = %s", (job_id, user_id)
        ).fetchone()
        if day is None:
            raise HTTPException(status_code=404, detail="weekly recap not generated yet")
        panels = conn.execute(
            "SELECT * FROM panels WHERE day_id = %s ORDER BY panel_id", (job_id,)
        ).fetchall()
        candidates = {
            c["panel_id"]: c
            for c in conn.execute(
                "SELECT * FROM candidates WHERE day_id = %s AND chosen", (job_id,)
            ).fetchall()
        }
        daily_rows = conn.execute(
            "SELECT job_id, date, mood FROM days WHERE user_id = %s AND date BETWEEN %s AND %s "
            "AND kind = 'daily' AND job_id != %s ORDER BY date",
            (user_id, monday, sunday, job_id),
        ).fetchall()
        week_day_ids = [job_id] + [r["job_id"] for r in daily_rows]
        pair_count = conn.execute(
            "SELECT count(*) AS n FROM image_pairs WHERE day_id = ANY(%s)", (week_day_ids,)
        ).fetchone()
        cast = conn.execute(
            "SELECT name FROM people WHERE user_id = %s ORDER BY created_at", (user_id,)
        ).fetchall()
        training_run = conn.execute(
            "SELECT id, kind, decision, created_at FROM training_runs "
            "ORDER BY created_at DESC LIMIT 1"
        ).fetchone()

    panel_list = [
        {
            "id": p["panel_id"],
            "caption": p["caption_a"],
            "url": candidates[p["panel_id"]]["url"] if p["panel_id"] in candidates else "",
        }
        for p in panels
    ]
    mood_series = [r["mood"] or "steady" for r in daily_rows]
    first_pick_pct = 0
    if candidates:
        first_pick_pct = round(
            100 * sum(1 for c in candidates.values() if c["chosen"]) / len(candidates)
        )

    return {
        "job_id": job_id,
        "week_label": iso_week,
        # en dash matches the frozen design's own date-range typography
        "date_range": f"{monday.isoformat()} – {sunday.isoformat()}",  # noqa: RUF001
        "user_label": user_id,
        "mood": day["mood"],
        "quiet_day": day["quiet_day"],
        "strip_url": day["strip_url"],
        "story_url": day["story_url"],
        "panels": panel_list,
        "mood_series": mood_series,
        "stats": {
            "days": len(daily_rows),
            "panels": len(panel_list),
            "pairs": pair_count["n"] if pair_count else 0,
            "first_pick_pct": first_pick_pct,
        },
        "cast": [{"name": r["name"]} for r in cast],
        "training_run": dict(training_run) if training_run else None,
    }


@router.get("/{iso_week}.pdf")
async def get_weekly_pdf(iso_week: str, user_id: str = Depends(current_user_id)) -> Response:
    # Registered BEFORE the plain `/{iso_week}` route below — Starlette
    # matches routes in registration order, and a bare `{iso_week}` path
    # param has no `.` restriction, so it would otherwise swallow
    # "2026-W38.pdf" whole as `iso_week` itself.
    ctx = _weekly_context(user_id, iso_week)
    template = _env.get_template("weekly.html")
    html = template.render(**ctx)

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        try:
            page = await browser.new_page()
            await page.set_content(html, wait_until="networkidle")
            pdf_bytes = await page.pdf(format="A4", print_background=True)
        finally:
            await browser.close()

    return Response(content=pdf_bytes, media_type="application/pdf")


@router.get("/{iso_week}")
async def get_weekly(iso_week: str, user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    ctx = _weekly_context(user_id, iso_week)
    ctx.pop("job_id")
    return ctx
