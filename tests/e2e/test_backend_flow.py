"""End-to-end backend flow: sign-in (mock) -> onboarding -> day ->
feedback -> weekly PDF -> export -> delete, all through the real API +
real Postgres + real Playwright PDF render.

This file runs with `services/api` as its `app` — the same constraint
`cast.py`'s module docstring documents rules out also importing
`services/worker`'s `app.worker` in the same process, so the "worker ran
and produced a real day/weekly recap" step is seeded directly via SQL
(the exact shape `app.persist.persist_day_state` writes — confirmed for
real, including the actual worker call, in `tests/unit/llm/
test_worker_jobs.py`, which runs with `services/worker` as its own
`app`). `/days/{date}/panels/{id}/regenerate` and `/account/export` are
asserted at "the API enqueued the right job with the right arguments"
(`tests/unit/api/test_feedback.py`/`test_account.py` already cover this
in isolation) — running the real worker functions for those specifically
isn't duplicated here for the same cross-package reason.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv
from fastapi.testclient import TestClient
from psycopg.rows import DictRow, dict_row
from storage import LocalFileStorage

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SCHEMA = "comiccanvas_test_e2e"
USER_ID = "e2e-user"
ISO_WEEK = "2026-W38"


def _database_url() -> str | None:
    load_dotenv(find_dotenv(usecwd=True))
    return os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    url = _database_url()
    if not url:
        pytest.skip("DATABASE_URL not configured; see the README for local Postgres setup")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')
    env = {**os.environ, "DB_SCHEMA": TEST_SCHEMA}
    subprocess.run(["uv", "run", "alembic", "upgrade", "head"], cwd=REPO_ROOT, env=env, check=True)
    yield url
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')


@pytest.fixture(autouse=True)
def _db_schema_env(monkeypatch: pytest.MonkeyPatch, database_url: str) -> None:
    monkeypatch.setenv("DB_SCHEMA", TEST_SCHEMA)
    monkeypatch.setenv("AUTH_MODE", "mock")


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalFileStorage:
    import app.deps as deps_module

    storage = LocalFileStorage(root=tmp_path, base_url="http://testserver")
    monkeypatch.setattr(deps_module, "get_storage", lambda: storage)
    return storage


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection[DictRow]]:
    connection = psycopg.connect(database_url, row_factory=dict_row)
    connection.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def arq_pool() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(arq_pool: AsyncMock) -> Iterator[TestClient]:
    from app.deps import arq_pool_dep
    from app.main import app

    app.dependency_overrides[arq_pool_dep] = lambda: arq_pool
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _auth(user_id: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {user_id}"}


def _seed_finished_day(
    conn: psycopg.Connection[DictRow], *, job_id: str, user_id: str, date: str
) -> None:
    """Stands in for `app.worker.run_daily_job`'s real persistence
    (verified separately, with the real graph, in `test_worker_jobs.py`)
    — writes the same shape `app.persist.persist_day_state` would."""
    conn.execute(
        "UPDATE jobs SET status = 'done', events = %s WHERE id = %s",
        (
            json.dumps([{"node": n} for n in ("input", "beats", "script", "generate", "compose")]),
            job_id,
        ),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source, mood, strip_url, story_url, cost_usd) "
        "VALUES (%s, %s, %s, 'text', 'content', %s, %s, 0.18)",
        (
            job_id,
            user_id,
            date,
            f"http://testserver/media/{job_id}/strip.png",
            f"http://testserver/media/{job_id}/story.png",
        ),
    )
    conn.execute(
        "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, importance, humor) "
        "VALUES (%s, 'b1', 'morning', 'kitchen', 'answered emails', 'content', 0.6, 0.2)",
        (job_id,),
    )
    for panel_id, (action, framing, caption) in enumerate(
        [
            ("typing on a laptop", "medium", "Finally answered that email."),
            ("pouring coffee", "close", "Coffee number two."),
            ("walking outdoors", "wide", "Took the long way home."),
            ("lifting a dumbbell", "medium", "Leg day."),
        ],
        start=1,
    ):
        conn.execute(
            "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, expression, "
            "action, framing, caption_a) VALUES (%s, %s, 'b1', 'kitchen', 'morning', "
            "'content', %s, %s, %s)",
            (job_id, panel_id, action, framing, caption),
        )
        candidate_id = f"{job_id}-c{panel_id}"
        candidate_url = f"http://testserver/c{panel_id}.png"
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
            "VALUES (%s, %s, %s, %s, %s, true)",
            (candidate_id, job_id, panel_id, candidate_url, panel_id),
        )
    conn.commit()


def test_backend_flow_sign_in_onboard_day_feedback_weekly_export_delete(
    client: TestClient, conn: psycopg.Connection[DictRow], arq_pool: AsyncMock
) -> None:
    headers = _auth(USER_ID)

    # --- sign in (mock) ---
    settings_response = client.get("/settings", headers=headers)
    assert settings_response.status_code == 200
    assert conn.execute("SELECT id FROM users WHERE id = %s", (USER_ID,)).fetchone() is not None

    # --- onboarding ---
    create_person = client.post("/cast", json={"name": "E2E Person"}, headers=headers)
    assert create_person.status_code == 201
    person_id = create_person.json()["id"]

    look_card = {
        "hair": "short black hair",
        "glasses": "none",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "mustard sweater",
        "distinguishing": "none",
        "gender_term": "person",
    }
    conn.execute(
        "INSERT INTO identity_models (person_id, master_path, look_card) VALUES (%s, %s, %s)",
        (person_id, f"people/{person_id}/master.png", json.dumps(look_card)),
    )
    conn.commit()

    # --- day ---
    create_day = client.post(
        "/days",
        json={"text": "Answered emails, made coffee, walked home, then hit the gym."},
        headers=headers,
    )
    assert create_day.status_code == 202
    job_id = create_day.json()["job_id"]
    arq_pool.enqueue_job.assert_awaited_once()
    assert arq_pool.enqueue_job.await_args.args[0] == "run_daily_job"

    _seed_finished_day(conn, job_id=job_id, user_id=USER_ID, date="2026-09-16")

    get_day = client.get("/days/2026-09-16", headers=headers)
    assert get_day.status_code == 200
    day_body = get_day.json()
    assert len(day_body["panels"]) == 4
    assert day_body["strip_url"] is not None

    # --- feedback ---
    thumb = client.post(
        "/feedback/thumb",
        json={"date": "2026-09-16", "panel_id": 1, "value": 1},
        headers=headers,
    )
    assert thumb.status_code == 201

    caption = client.post(
        "/feedback/caption",
        json={
            "date": "2026-09-16",
            "panel_id": 1,
            "original": "x",
            "edited": "Answered email #1.",
        },
        headers=headers,
    )
    assert caption.status_code == 201

    arq_pool.enqueue_job.reset_mock()
    regenerate = client.post(
        "/days/2026-09-16/panels/1/regenerate", json={"note": "try again"}, headers=headers
    )
    assert regenerate.status_code == 202
    arq_pool.enqueue_job.assert_awaited_once_with(
        "regenerate_panel_job", job_id, USER_ID, 1, "try again"
    )

    # --- weekly recap + PDF ---
    weekly_job_id = f"{USER_ID}:weekly:{ISO_WEEK}"
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'weekly', 'pending')",
        (weekly_job_id, USER_ID),
    )
    conn.commit()
    _seed_finished_day(conn, job_id=weekly_job_id, user_id=USER_ID, date="2026-09-20")

    get_weekly = client.get(f"/weekly/{ISO_WEEK}", headers=headers)
    assert get_weekly.status_code == 200
    assert len(get_weekly.json()["panels"]) == 4

    weekly_pdf = client.get(f"/weekly/{ISO_WEEK}.pdf", headers=headers)
    assert weekly_pdf.status_code == 200
    assert weekly_pdf.headers["content-type"] == "application/pdf"
    assert weekly_pdf.content.startswith(b"%PDF")

    # --- export ---
    arq_pool.enqueue_job.reset_mock()
    export_response = client.get("/account/export", headers=headers)
    assert export_response.status_code == 200
    export_job_id = export_response.json()["job_id"]
    arq_pool.enqueue_job.assert_awaited_once_with("export_account_job", export_job_id, USER_ID)
    export_job = conn.execute(
        "SELECT kind, status FROM jobs WHERE id = %s", (export_job_id,)
    ).fetchone()
    assert export_job is not None
    assert export_job["kind"] == "export"

    # --- delete ---
    delete_response = client.delete("/account", headers=headers)
    assert delete_response.status_code == 200
    assert conn.execute("SELECT id FROM users WHERE id = %s", (USER_ID,)).fetchone() is None
    assert conn.execute("SELECT id FROM people WHERE id = %s", (person_id,)).fetchone() is None
