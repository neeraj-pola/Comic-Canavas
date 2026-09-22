"""`POST /days` enqueues, `GET /days/{date}` reads back what the worker
would have persisted (seeded directly here — `run_daily_job` itself is
covered end-to-end in `tests/unit/llm/test_worker_jobs.py`; this file
tests the API's own read/write wiring, not the pipeline)."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket


def test_create_day_enqueues_run_daily_job(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    response = client.post(
        "/days", json={"text": "A quiet day."}, headers=auth_headers("days-user")
    )

    assert response.status_code == 202
    job_id = response.json()["job_id"]
    assert response.json()["status"] == "pending"
    arq_pool.enqueue_job.assert_awaited_once()
    args = arq_pool.enqueue_job.await_args.args
    assert args[0] == "run_daily_job"
    assert args[1] == job_id

    row = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    assert row["status"] == "pending"


def test_create_day_is_idempotent_while_a_job_is_active(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]], arq_pool: AsyncMock
) -> None:
    headers = auth_headers("days-user-2")
    first = client.post("/days", json={"text": "hello"}, headers=headers)
    second = client.post("/days", json={"text": "hello again"}, headers=headers)

    assert first.json()["job_id"] == second.json()["job_id"]
    arq_pool.enqueue_job.assert_awaited_once()  # not re-enqueued the second time


def test_create_day_re_enqueues_a_fresh_run_after_the_previous_one_is_done(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """"Redo today's strip" calls this same endpoint again for an already-`done` day — only
    "pending"/"running" are genuinely in-flight, so "done" must fall through to a fresh enqueue
    rather than returning the stale job_id/status."""
    headers = auth_headers("days-user-redo")
    first = client.post("/days", json={"text": "first pass"}, headers=headers)
    job_id = first.json()["job_id"]
    conn.execute("UPDATE jobs SET status = 'done' WHERE id = %s", (job_id,))
    conn.commit()

    second = client.post("/days", json={"text": "redo please"}, headers=headers)

    assert second.json()["job_id"] == job_id
    assert second.json()["status"] == "pending"
    assert arq_pool.enqueue_job.await_count == 2  # re-enqueued, not skipped
    row = conn.execute("SELECT status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    assert row["status"] == "pending"


def test_create_day_422s_without_text_or_audio_url(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.post("/days", json={}, headers=auth_headers("days-user-3"))
    assert response.status_code == 422


def test_get_day_returns_persisted_panels_and_candidates(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    user_id, job_id = "days-user-4", "days-job-4"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user_id,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, mood, strip_url, cost_usd) "
        "VALUES (%s, %s, '2026-09-16', 'text', 'content', 'http://x/strip.png', 0.12)",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES (%s, 'b1', 'evening', 'gym', 'lifted weights', 'content', 0.6, 0.2)",
        (job_id,),
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, expression, "
        "action, framing, caption_a) VALUES "
        "(%s, 1, 'b1', 'gym', 'evening', 'content', 'lifting', 'wide', 'leg day')",
        (job_id,),
    )
    conn.execute(
        "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
        "VALUES ('c1', %s, 1, 'http://x/c1.png', 42, true)",
        (job_id,),
    )
    conn.commit()

    response = client.get("/days/2026-09-16", headers=auth_headers(user_id))

    assert response.status_code == 200
    body = response.json()
    assert body["mood"] == "content"
    assert body["cost_usd"] == 0.12
    assert len(body["panels"]) == 1
    assert body["panels"][0]["caption_a"] == "leg day"
    assert len(body["candidates"]) == 1
    assert body["candidates"][0]["chosen"] is True
    assert body["prompts"] == []


def test_get_day_returns_persisted_image_prompts_linked_to_their_candidate(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    """Confirms the `image_prompts` read path for a day that actually has one."""
    user_id, job_id = "days-user-5", "days-job-5"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user_id,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, mood, strip_url, cost_usd) "
        "VALUES (%s, %s, '2026-09-16', 'text', 'content', 'http://x/strip.png', 0.12)",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES (%s, 'b1', 'evening', 'gym', 'lifted weights', 'content', 0.6, 0.2)",
        (job_id,),
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, expression, "
        "action, framing, caption_a) VALUES "
        "(%s, 1, 'b1', 'gym', 'evening', 'content', 'lifting', 'wide', 'leg day')",
        (job_id,),
    )
    conn.execute(
        "INSERT INTO image_prompts (id, day_id, panel_id, generator, character_clause, "
        "environment_clause, positive, negative, seed, guidance, steps) VALUES "
        "('days-job-5-p1', %s, 1, 'mock', 'a person', 'a gym', 'a person, a gym', "
        "'blurry', 4176670031, 3.5, 28)",
        (job_id,),
    )
    conn.execute(
        "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen, prompt_id) "
        "VALUES ('c-days-5', %s, 1, 'http://x/c1.png', 42, true, 'days-job-5-p1')",
        (job_id,),
    )
    conn.commit()

    response = client.get("/days/2026-09-16", headers=auth_headers(user_id))

    assert response.status_code == 200
    body = response.json()
    assert len(body["prompts"]) == 1
    prompt = body["prompts"][0]
    assert prompt["id"] == "days-job-5-p1"
    assert prompt["panel_id"] == 1
    assert prompt["positive"] == "a person, a gym"
    assert prompt["seed"] == 4176670031
    assert body["candidates"][0]["prompt_id"] == "days-job-5-p1"


def test_get_day_404s_for_a_date_with_no_day(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.get("/days/2020-01-01", headers=auth_headers("days-user-5"))
    assert response.status_code == 404


def test_create_day_accepts_a_past_date_for_backfilling(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """Backfills a whole week for the weekly recap. The date is optional (defaults to today) and
    names the job and the day."""
    headers = auth_headers("days-user-backfill")

    response = client.post("/days", json={"text": "Monday.", "date": "2026-09-14"}, headers=headers)

    assert response.status_code == 202
    assert response.json()["job_id"] == "days-user-backfill:2026-09-14"
    args = arq_pool.enqueue_job.await_args.args
    assert (args[0], args[1], args[3]) == (
        "run_daily_job",
        "days-user-backfill:2026-09-14",
        "2026-09-14",
    )


def test_create_day_rejects_a_future_or_malformed_date(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]], arq_pool: AsyncMock
) -> None:
    from datetime import date, timedelta

    headers = auth_headers("days-user-baddate")
    tomorrow = (date.today() + timedelta(days=1)).isoformat()

    future = client.post("/days", json={"text": "x", "date": tomorrow}, headers=headers)
    garbage = client.post("/days", json={"text": "x", "date": "last tuesday"}, headers=headers)

    assert future.status_code == 422 and "future" in future.json()["detail"]
    assert garbage.status_code == 422
    arq_pool.enqueue_job.assert_not_called()


def test_a_weekly_recap_row_never_shows_up_as_a_diary_entry(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    """The recap is stored as a `days` row dated the week's Sunday, next to
    the person's real Sunday entry. Reading a day, the library and search
    must only ever see the real (`kind = 'daily'`) one."""
    user = "days-user-weekly-kind"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user,))
    for job_id, kind, mood in (
        (f"{user}:2026-09-20", "daily", "my sunday"),
        (f"{user}:weekly:2026-W38", "weekly", "week in review"),
    ):
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, %s, 'done')",
            (job_id, user, kind),
        )
        conn.execute(
            "INSERT INTO days (job_id, user_id, date, source, mood, kind) "
            "VALUES (%s, %s, '2026-09-20', 'text', %s, %s)",
            (job_id, user, mood, kind),
        )
    conn.commit()
    headers = auth_headers(user)

    day = client.get("/days/2026-09-20", headers=headers)
    library = client.get("/library?month=2026-09", headers=headers)

    assert day.status_code == 200 and day.json()["mood"] == "my sunday"
    assert [d["mood"] for d in library.json()["days"]] == ["my sunday"]


def test_get_day_returns_the_picks_and_ratings_already_given(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    """The latest pick per panel and the latest rating per image come back, so going back to a
    day shows what was already decided rather than an untouched picker."""
    user_id, job_id = "days-user-saved", "days-job-saved"
    conn.execute("INSERT INTO users (id) VALUES (%s)", (user_id,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES (%s, %s, '2026-09-16', 'text')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, created_at) VALUES "
        "(%s, 1, 'a', 'b', 'ab', now() - interval '2 minutes'), "
        "(%s, 1, 'b', 'a', 'ab', now() - interval '1 minute'), "
        "(%s, 2, 'c', 'd', 'ab', now())",
        (job_id, job_id, job_id),
    )
    conn.execute(
        "INSERT INTO image_ratings (day_id, panel_id, url, rating, created_at) VALUES "
        "(%s, 1, 'b', 0, now() - interval '2 minutes'), (%s, 1, 'b', 2, now())",
        (job_id, job_id),
    )
    conn.commit()

    body = client.get("/days/2026-09-16", headers=auth_headers(user_id)).json()

    assert body["picks"] == {"1": "b", "2": "c"}  # the changed mind wins
    assert body["ratings"] == {"b": 2}
