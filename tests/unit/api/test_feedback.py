"""`POST /feedback/{pair,caption,thumb}` and
`POST /days/{date}/panels/{id}/regenerate`."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow

pytestmark = pytest.mark.enable_socket


def _seed_day(conn: psycopg.Connection[DictRow], *, user_id: str, job_id: str) -> None:
    conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES (%s, %s, '2026-09-16', 'text')",
        (job_id, user_id),
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, place, time_of_day, expression, action, "
        "framing, caption_a) VALUES (%s, 1, 'gym', 'evening', 'content', 'lifting', "
        "'wide', 'leg day')",
        (job_id,),
    )
    conn.commit()


def test_feedback_pair_writes_a_row(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    _seed_day(conn, user_id="fb-user", job_id="fb-job-1")

    response = client.post(
        "/feedback/pair",
        json={
            "date": "2026-09-16",
            "panel_id": 1,
            "chosen": "a.png",
            "rejected": "b.png",
            "source": "ab",
        },
        headers=auth_headers("fb-user"),
    )

    assert response.status_code == 201
    row = conn.execute("SELECT chosen FROM image_pairs WHERE day_id = 'fb-job-1'").fetchone()
    assert row is not None
    assert row["chosen"] == "a.png"


def test_feedback_pick_writes_a_row_per_option_passed_on_under_one_tap_and_enqueues_the_pick(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """Best-of-3: one tap = the favourite of a panel's options."""
    _seed_day(conn, user_id="fb-user-pick", job_id="fb-job-pick")

    response = client.post(
        "/feedback/pick",
        json={
            "date": "2026-09-16",
            "panel_id": 1,
            "picked": "b.png",
            "others": ["a.png", "c.png", "a.png"],  # a repeat is ignored
        },
        headers=auth_headers("fb-user-pick"),
    )

    assert response.status_code == 201
    rows = conn.execute(
        "SELECT chosen, rejected, source, tap_id FROM image_pairs WHERE day_id = 'fb-job-pick' "
        "ORDER BY id"
    ).fetchall()
    assert [(r["chosen"], r["rejected"], r["source"]) for r in rows] == [
        ("b.png", "a.png", "ab"),
        ("b.png", "c.png", "ab"),
    ]
    assert len({r["tap_id"] for r in rows}) == 1 and rows[0]["tap_id"] == response.json()["tap_id"]
    arq_pool.enqueue_job.assert_awaited_once_with(
        "apply_pair_choice_job", "fb-job-pick", "fb-user-pick", 1, "b.png"
    )


def test_feedback_pick_rejects_passing_on_the_image_you_picked(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    _seed_day(conn, user_id="fb-user-pick2", job_id="fb-job-pick2")
    response = client.post(
        "/feedback/pick",
        json={"date": "2026-09-16", "panel_id": 1, "picked": "a.png", "others": ["a.png"]},
        headers=auth_headers("fb-user-pick2"),
    )
    assert response.status_code == 422
    arq_pool.enqueue_job.assert_not_called()


def test_feedback_caption_writes_a_row_and_updates_the_panel(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    _seed_day(conn, user_id="fb-user-2", job_id="fb-job-2")

    response = client.post(
        "/feedback/caption",
        json={"date": "2026-09-16", "panel_id": 1, "original": "leg day", "edited": "leg day!"},
        headers=auth_headers("fb-user-2"),
    )

    assert response.status_code == 201
    panel = conn.execute(
        "SELECT caption_a FROM panels WHERE day_id = 'fb-job-2' AND panel_id = 1"
    ).fetchone()
    assert panel is not None
    assert panel["caption_a"] == "leg day!"
    # Saving a caption must also re-compose the strip so the download reflects it.
    assert response.json()["caption"] == "leg day!"
    args = arq_pool.enqueue_job.await_args.args
    assert (args[0], args[1], args[2]) == ("recompose_strip_job", "fb-job-2", "fb-user-2")


def test_feedback_caption_rejects_empty_and_overlong_text(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    _seed_day(conn, user_id="fb-user-2b", job_id="fb-job-2b")
    headers = auth_headers("fb-user-2b")
    body = {"date": "2026-09-16", "panel_id": 1, "original": "leg day"}

    empty = client.post("/feedback/caption", json={**body, "edited": "   "}, headers=headers)
    too_long = client.post("/feedback/caption", json={**body, "edited": "x" * 61}, headers=headers)
    exactly_60 = client.post(
        "/feedback/caption", json={**body, "edited": "y" * 60}, headers=headers
    )

    assert empty.status_code == 422
    assert too_long.status_code == 422
    assert exactly_60.status_code == 201
    assert arq_pool.enqueue_job.await_count == 1  # only the valid edit re-composes


def test_regenerate_passes_the_note_to_the_worker_and_caps_its_length(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    _seed_day(conn, user_id="fb-user-2c", job_id="fb-job-2c")
    headers = auth_headers("fb-user-2c")

    ok = client.post(
        "/days/2026-09-16/panels/1/regenerate", json={"note": "look less tired"}, headers=headers
    )
    too_long = client.post(
        "/days/2026-09-16/panels/1/regenerate", json={"note": "n" * 301}, headers=headers
    )

    assert ok.status_code == 202
    assert arq_pool.enqueue_job.await_args.args[-1] == "look less tired"
    assert too_long.status_code == 422


def test_feedback_thumb_writes_a_row(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
) -> None:
    _seed_day(conn, user_id="fb-user-3", job_id="fb-job-3")

    response = client.post(
        "/feedback/thumb",
        json={"date": "2026-09-16", "panel_id": 1, "value": 1},
        headers=auth_headers("fb-user-3"),
    )

    assert response.status_code == 201
    row = conn.execute("SELECT value FROM thumbs WHERE day_id = 'fb-job-3'").fetchone()
    assert row is not None
    assert row["value"] == 1


def test_feedback_pair_with_ab_source_enqueues_apply_pair_choice_job(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """`POST /feedback/pair` with `source="ab"` must also enqueue the real recompose job, so the
    displayed strip actually switches to the tapped image."""
    _seed_day(conn, user_id="fb-user-6", job_id="fb-job-6")

    response = client.post(
        "/feedback/pair",
        json={
            "date": "2026-09-16",
            "panel_id": 1,
            "chosen": "a.png",
            "rejected": "b.png",
            "source": "ab",
        },
        headers=auth_headers("fb-user-6"),
    )

    assert response.status_code == 201
    arq_pool.enqueue_job.assert_awaited_once_with(
        "apply_pair_choice_job", "fb-job-6", "fb-user-6", 1, "a.png"
    )


def test_feedback_pair_with_regenerate_source_does_not_enqueue_apply_pair_choice_job(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """`source="regenerate"` rows are written directly by `regenerate_
    panel_job` itself (never through this route in real usage), and that
    job already recomposes as part of its own retry machinery — enqueuing
    a second recompose here would be redundant, not just harmless."""
    _seed_day(conn, user_id="fb-user-7", job_id="fb-job-7")

    response = client.post(
        "/feedback/pair",
        json={
            "date": "2026-09-16",
            "panel_id": 1,
            "chosen": "a.png",
            "rejected": "b.png",
            "source": "regenerate",
        },
        headers=auth_headers("fb-user-7"),
    )

    assert response.status_code == 201
    arq_pool.enqueue_job.assert_not_awaited()


def test_feedback_pair_404s_for_an_unknown_date(
    client: TestClient, auth_headers: Callable[[str], dict[str, str]]
) -> None:
    response = client.post(
        "/feedback/pair",
        json={"date": "2020-01-01", "panel_id": 1, "chosen": "a", "rejected": "b", "source": "ab"},
        headers=auth_headers("fb-user-4"),
    )
    assert response.status_code == 404


def test_regenerate_enqueues_the_worker_job(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    _seed_day(conn, user_id="fb-user-5", job_id="fb-job-5")

    response = client.post(
        "/days/2026-09-16/panels/1/regenerate",
        json={"note": "try a different pose"},
        headers=auth_headers("fb-user-5"),
    )

    assert response.status_code == 202
    arq_pool.enqueue_job.assert_awaited_once_with(
        "regenerate_panel_job", "fb-job-5", "fb-user-5", 1, "try a different pose"
    )


def test_feedback_rating_records_it_and_asks_the_worker_to_relearn(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    """Rate a single image off / ok / great."""
    _seed_day(conn, user_id="fb-user-rate", job_id="fb-job-rate")
    conn.execute(
        "INSERT INTO candidates (id, day_id, panel_id, url, seed) "
        "VALUES ('c1', 'fb-job-rate', 1, 'http://x/a.png', 1)"
    )
    conn.commit()
    headers = auth_headers("fb-user-rate")

    ok = client.post(
        "/feedback/rating",
        json={"date": "2026-09-16", "panel_id": 1, "url": "http://x/a.png", "rating": 2},
        headers=headers,
    )

    assert ok.status_code == 201
    row = conn.execute(
        "SELECT rating, url FROM image_ratings WHERE day_id = 'fb-job-rate'"
    ).fetchone()
    assert row is not None and (row["rating"], row["url"]) == (2, "http://x/a.png")
    arq_pool.enqueue_job.assert_awaited_once_with("rebuild_preference_job", "fb-user-rate")

    unknown = client.post(
        "/feedback/rating",
        json={"date": "2026-09-16", "panel_id": 1, "url": "http://x/none.png", "rating": 1},
        headers=headers,
    )
    assert unknown.status_code == 404
    bad = client.post(
        "/feedback/rating",
        json={"date": "2026-09-16", "panel_id": 1, "url": "http://x/a.png", "rating": 3},
        headers=headers,
    )
    assert bad.status_code == 422
