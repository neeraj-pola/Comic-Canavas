"""`GET /weekly/{iso_week}` (recap/mood/stats/cast) and
`GET /weekly/{iso_week}.pdf` (real Playwright render). Seeds the weekly
`days`/`panels`/`candidates` rows directly (the same shape
`app.worker.run_weekly_job` would persist via `persist_day_state`) —
`run_weekly_job` itself is a worker-side concern, not retested here."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket

ISO_WEEK = "2026-W38"  # 2026-09-14 (Mon) .. 2026-09-20 (Sun)


def _seed_week(conn: psycopg.Connection[DictRow], *, user_id: str) -> str:
    weekly_job_id = f"{user_id}:weekly:{ISO_WEEK}"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user_id,))

    daily_job_id = f"{user_id}:daily:2026-09-15"
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (daily_job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, mood) "
        "VALUES (%s, %s, '2026-09-15', 'text', 'content')",
        (daily_job_id, user_id),
    )

    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'weekly', 'done')",
        (weekly_job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, strip_url, story_url) "
        "VALUES (%s, %s, '2026-09-20', 'text', 'http://x/strip.png', 'http://x/story.png')",
        (weekly_job_id, user_id),
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, place, time_of_day, expression, action, "
        "framing, caption_a) VALUES (%s, 1, 'gym', 'evening', 'content', 'lifting', "
        "'wide', 'Leg day.')",
        (weekly_job_id,),
    )
    conn.execute(
        "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
        "VALUES ('wc1', %s, 1, 'http://x/wc1.png', 1, true)",
        (weekly_job_id,),
    )
    conn.commit()
    return weekly_job_id


def test_generate_weekly_enqueues_the_worker_job(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]], arq_pool: AsyncMock
) -> None:
    response = client.post(f"/weekly/{ISO_WEEK}/generate", headers=auth_headers("weekly-user"))

    assert response.status_code == 202
    arq_pool.enqueue_job.assert_awaited_once_with(
        "run_weekly_job", f"weekly-user:weekly:{ISO_WEEK}", "weekly-user", ISO_WEEK
    )


def test_generate_weekly_rejects_a_malformed_week(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.post("/weekly/not-a-week/generate", headers=auth_headers("weekly-user-2"))
    assert response.status_code == 422


def test_get_weekly_404s_before_generation(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.get(f"/weekly/{ISO_WEEK}", headers=auth_headers("weekly-user-3"))
    assert response.status_code == 404


def test_get_weekly_returns_recap_mood_stats_and_cast(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user_id = "weekly-user-4"
    _seed_week(conn, user_id=user_id)
    conn.execute("INSERT INTO people (id, user_id, name) VALUES ('p1', %s, 'You')", (user_id,))
    conn.commit()

    response = client.get(f"/weekly/{ISO_WEEK}", headers=auth_headers(user_id))

    assert response.status_code == 200
    body = response.json()
    assert body["strip_url"] == "http://x/strip.png"
    assert len(body["panels"]) == 1
    assert body["panels"][0]["caption"] == "Leg day."
    assert body["mood_series"] == ["content"]
    assert body["stats"]["days"] == 1
    assert body["stats"]["panels"] == 1
    assert body["cast"] == [{"name": "You"}]


def test_get_weekly_pdf_renders_a_real_pdf(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user_id = "weekly-user-5"
    _seed_week(conn, user_id=user_id)

    response = client.get(f"/weekly/{ISO_WEEK}.pdf", headers=auth_headers(user_id))

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/pdf"
    assert response.content.startswith(b"%PDF")
