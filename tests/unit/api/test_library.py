"""`GET /library?month=`, `GET /search?q=` against the real relational
tables (search is lexical ILIKE — see `app/routers/library.py`'s module
docstring for why)."""

from __future__ import annotations

from collections.abc import Callable

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket


def _seed_day_with_beat(
    conn: psycopg.Connection[DictRow],
    *,
    user_id: str,
    job_id: str,
    date: str,
    event: str,
    place: str,
    mood: str = "content",
) -> None:
    conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, mood, strip_url) "
        "VALUES (%s, %s, %s, 'text', %s, %s)",
        (job_id, user_id, date, mood, f"http://localhost:8000/media/{job_id}/strip.png"),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES (%s, 'b1', 'evening', %s, %s, 'content', 0.6, 0.2)",
        (job_id, place, event),
    )
    conn.commit()


def test_library_returns_days_in_the_given_month(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    _seed_day_with_beat(
        conn, user_id="lib-user", job_id="lib-job-1", date="2026-09-03", event="gym", place="gym"
    )
    _seed_day_with_beat(
        conn,
        user_id="lib-user",
        job_id="lib-job-2",
        date="2026-08-15",
        event="reading",
        place="bedroom",
    )

    response = client.get("/library", params={"month": "2026-09"}, headers=auth_headers("lib-user"))

    assert response.status_code == 200
    dates = [d["date"] for d in response.json()["days"]]
    assert dates == ["2026-09-03"]


def test_search_finds_the_gym_beat_by_event_text(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    _seed_day_with_beat(
        conn,
        user_id="search-user",
        job_id="search-job-1",
        date="2026-09-03",
        event="first gym session in two weeks",
        place="gym",
    )
    _seed_day_with_beat(
        conn,
        user_id="search-user",
        job_id="search-job-2",
        date="2026-09-04",
        event="quiet reading afternoon",
        place="bedroom",
    )

    response = client.get("/search", params={"q": "gym"}, headers=auth_headers("search-user"))

    assert response.status_code == 200
    hits = response.json()["hits"]
    assert len(hits) == 1
    assert hits[0]["date"] == "2026-09-03"
