"""The four cron entrypoints, unit-tested with frozen time — each takes
`now: datetime` as an explicit argument (see `app/cron.py`'s module
docstring for why), so "frozen time" just means constructing the exact
`datetime` a test wants; no time-mocking library needed.

`remind`/`weekly_recap`/`export_feedback` touch a real Postgres
(`enable_socket`), isolated in `comiccanvas_test_scheduler` via
`DB_SCHEMA` + a real `alembic upgrade head` (same mechanism
`tests/integration/test_migrations.py` established).
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv
from psycopg.rows import DictRow, dict_row

from app.cron import (
    MockNotificationAdapter,
    export_feedback,
    held_out_weeks,
    is_export_feedback_time,
    is_kick_training_time,
    is_rollback_needed,
    is_weekly_recap_time,
    kick_training,
    live_rollback,
    remind,
    weekly_recap,
)

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "comiccanvas_test_scheduler"

# A real Sunday (2026-09-20 is a Sunday) at each cron's exact minute.
WEEKLY_RECAP_AT = datetime(2026, 9, 20, 1, 0, tzinfo=UTC)
EXPORT_FEEDBACK_AT = datetime(2026, 9, 20, 1, 30, tzinfo=UTC)
KICK_TRAINING_AT = datetime(2026, 9, 20, 2, 0, tzinfo=UTC)
NOT_DUE = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)  # a Monday noon


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


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection[DictRow]]:
    connection = psycopg.connect(database_url, row_factory=dict_row)
    connection.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
    connection.execute("TRUNCATE users, settings, days, jobs, training_runs CASCADE")
    connection.commit()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


def test_is_weekly_recap_time_only_matches_sunday_0100() -> None:
    assert is_weekly_recap_time(WEEKLY_RECAP_AT)
    assert not is_weekly_recap_time(NOT_DUE)
    assert not is_weekly_recap_time(EXPORT_FEEDBACK_AT)


def test_is_export_feedback_time_only_matches_sunday_0130() -> None:
    assert is_export_feedback_time(EXPORT_FEEDBACK_AT)
    assert not is_export_feedback_time(WEEKLY_RECAP_AT)


def test_is_kick_training_time_only_matches_sunday_0200() -> None:
    assert is_kick_training_time(KICK_TRAINING_AT)
    assert not is_kick_training_time(EXPORT_FEEDBACK_AT)


def test_remind_sends_to_users_whose_reminder_time_matches_now(
    conn: psycopg.Connection[DictRow],
) -> None:
    conn.execute("INSERT INTO users (id) VALUES ('u1'), ('u2')")
    conn.execute(
        "INSERT INTO settings (user_id, reminder_time) VALUES ('u1', '09:00'), ('u2', '21:00')"
    )
    adapter = MockNotificationAdapter()

    due = remind(datetime(2026, 9, 16, 9, 0, tzinfo=UTC), conn=conn, adapter=adapter)

    assert due == ["u1"]
    assert adapter.sent == [("u1", "Time for today's comic!")]


def test_remind_sends_to_nobody_when_no_time_matches(conn: psycopg.Connection[DictRow]) -> None:
    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    conn.execute("INSERT INTO settings (user_id, reminder_time) VALUES ('u1', '09:00')")
    adapter = MockNotificationAdapter()

    due = remind(NOT_DUE, conn=conn, adapter=adapter)

    assert due == []
    assert adapter.sent == []


async def test_weekly_recap_noops_outside_its_scheduled_minute(
    conn: psycopg.Connection[DictRow],
) -> None:
    arq_pool = AsyncMock()
    result = await weekly_recap(NOT_DUE, conn=conn, arq_pool=arq_pool)
    assert result == []
    arq_pool.enqueue_job.assert_not_called()


async def test_weekly_recap_enqueues_a_job_per_user_with_a_recorded_day(
    conn: psycopg.Connection[DictRow],
) -> None:
    conn.execute("INSERT INTO users (id) VALUES ('u1'), ('u2')")
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j1', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES ('j1', 'u1', '2026-09-14', 'text')"
    )
    arq_pool = AsyncMock()

    result = await weekly_recap(WEEKLY_RECAP_AT, conn=conn, arq_pool=arq_pool)

    assert result == ["u1"]
    arq_pool.enqueue_job.assert_awaited_once_with(
        "run_weekly_job", "u1:weekly:2026-W38", "u1", "2026-W38"
    )


def test_export_feedback_noops_outside_its_scheduled_minute(
    conn: psycopg.Connection[DictRow],
) -> None:
    assert export_feedback(NOT_DUE, conn=conn) is None


def test_export_feedback_writes_a_snapshot_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection[DictRow]
) -> None:
    import app.cron as cron_module

    monkeypatch.setattr(cron_module, "_DATA_ROOT", tmp_path)
    # Isolate from the real, frozen `ml/goldens/pairs/heldout_week.jsonl` —
    # this test's own fixture date must not accidentally collide with
    # whatever real week that file happens to cover.
    monkeypatch.setattr(cron_module, "_HELDOUT_PATH", tmp_path / "no-such-file.jsonl")
    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j1', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES ('j1', 'u1', '2026-09-14', 'text')"
    )
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source) "
        "VALUES ('j1', 1, 'a.png', 'b.png', 'ab')"
    )

    snapshot_hash = export_feedback(EXPORT_FEEDBACK_AT, conn=conn)

    assert snapshot_hash is not None
    out_file = tmp_path / "2026-W38" / "feedback.json"
    assert out_file.is_file()
    row = conn.execute(
        "SELECT snapshot_hash FROM training_runs WHERE kind = 'export_feedback'"
    ).fetchone()
    assert row is not None
    assert row["snapshot_hash"] == snapshot_hash


async def test_kick_training_fires_only_on_schedule(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mocks the real training call (`ml.reward_model.train.main`) —
    running it for real here would hit the real dev DB and re-push a file
    to the real HF Hub repo on every test run, which tests must never do."""
    monkeypatch.setattr("ml.reward_model.train.main", lambda: None)

    assert await kick_training(KICK_TRAINING_AT) is True
    assert await kick_training(NOT_DUE) is False


async def test_kick_training_handles_a_missing_heldout_file_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise() -> None:
        raise FileNotFoundError

    monkeypatch.setattr("ml.reward_model.train.main", _raise)

    assert await kick_training(KICK_TRAINING_AT) is True


# --- held-out week exclusion ---


def test_held_out_weeks_reads_the_real_iso_weeks_from_a_jsonl_file(tmp_path: Path) -> None:
    path = tmp_path / "heldout.jsonl"
    path.write_text('{"date": "2026-09-14"}\n{"date": "2026-09-20"}\n{"date": "2026-08-01"}\n')

    assert held_out_weeks(path) == {"2026-W38", "2026-W31"}


def test_held_out_weeks_is_empty_when_the_file_does_not_exist(tmp_path: Path) -> None:
    assert held_out_weeks(tmp_path / "does-not-exist.jsonl") == set()


def test_export_feedback_excludes_rows_from_the_held_out_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection[DictRow]
) -> None:
    import app.cron as cron_module

    monkeypatch.setattr(cron_module, "_DATA_ROOT", tmp_path)
    heldout_path = tmp_path / "heldout.jsonl"
    heldout_path.write_text('{"date": "2026-09-14"}\n')  # 2026-W38
    monkeypatch.setattr(cron_module, "_HELDOUT_PATH", heldout_path)

    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    # Held-out day (2026-09-14, in 2026-W38).
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j-heldout', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) "
        "VALUES ('j-heldout', 'u1', '2026-09-14', 'text')"
    )
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source) "
        "VALUES ('j-heldout', 1, 'a.png', 'b.png', 'ab')"
    )
    # A different, non-held-out day.
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j-real', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) "
        "VALUES ('j-real', 'u1', '2026-10-01', 'text')"
    )
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source) "
        "VALUES ('j-real', 1, 'c.png', 'd.png', 'ab')"
    )

    snapshot_hash = export_feedback(EXPORT_FEEDBACK_AT, conn=conn)

    assert snapshot_hash is not None
    exported = json.loads((tmp_path / "2026-W38" / "feedback.json").read_text())
    assert [p["day_id"] for p in exported["image_pairs"]] == ["j-real"]


def test_export_feedback_returns_none_when_everything_is_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, conn: psycopg.Connection[DictRow]
) -> None:
    import app.cron as cron_module

    monkeypatch.setattr(cron_module, "_DATA_ROOT", tmp_path)
    heldout_path = tmp_path / "heldout.jsonl"
    heldout_path.write_text('{"date": "2026-09-14"}\n')
    monkeypatch.setattr(cron_module, "_HELDOUT_PATH", heldout_path)

    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j-heldout', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) "
        "VALUES ('j-heldout', 'u1', '2026-09-14', 'text')"
    )
    conn.execute(
        "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source) "
        "VALUES ('j-heldout', 1, 'a.png', 'b.png', 'ab')"
    )

    assert export_feedback(EXPORT_FEEDBACK_AT, conn=conn) is None


# --- live rollback ---


def test_is_rollback_needed_true_when_drop_exceeds_the_threshold() -> None:
    assert is_rollback_needed([0.60, 0.55, 0.50], baseline_rate=0.70) is True


def test_is_rollback_needed_false_when_drop_is_within_the_threshold() -> None:
    assert is_rollback_needed([0.68, 0.67, 0.69], baseline_rate=0.70) is False


def test_is_rollback_needed_false_with_no_recent_data() -> None:
    assert is_rollback_needed([], baseline_rate=0.70) is False


def test_live_rollback_alerts_when_first_pick_rate_really_dropped(
    conn: psycopg.Connection[DictRow],
) -> None:
    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    for i in range(3):  # 3 days, every panel's FIRST-created candidate rejected
        job_id = f"j{i}"
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, 'u1', 'daily', 'done')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO days (job_id, user_id, date, source) VALUES (%s, 'u1', %s, 'text')",
            (job_id, f"2026-09-{14 + i:02d}"),
        )
        conn.execute(
            "INSERT INTO panels (day_id, panel_id, place, time_of_day, expression, "
            "action, framing, caption_a) VALUES (%s, 1, 'gym', 'evening', 'content', "
            "'lifting', 'wide', 'leg day')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen, created_at) "
            "VALUES (%s, %s, 1, 'a.png', 1, false, now() - interval '1 second')",
            (f"{job_id}-c1", job_id),
        )
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
            "VALUES (%s, %s, 1, 'b.png', 2, true)",
            (f"{job_id}-c2", job_id),
        )
    conn.commit()

    adapter = MockNotificationAdapter()
    triggered = live_rollback(conn=conn, user_id="u1", baseline_rate=0.90, adapter=adapter)

    assert triggered is True
    assert len(adapter.sent) == 1
    assert adapter.sent[0][0] == "u1"
    assert "ALERT" in adapter.sent[0][1]


def test_live_rollback_does_not_alert_when_rate_is_healthy(
    conn: psycopg.Connection[DictRow],
) -> None:
    conn.execute("INSERT INTO users (id) VALUES ('u1')")
    conn.execute(
        "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j1', 'u1', 'daily', 'done')"
    )
    conn.execute(
        "INSERT INTO days (job_id, user_id, date, source) VALUES ('j1', 'u1', '2026-09-14', 'text')"
    )
    conn.execute(
        "INSERT INTO panels (day_id, panel_id, place, time_of_day, expression, "
        "action, framing, caption_a) VALUES ('j1', 1, 'gym', 'evening', 'content', "
        "'lifting', 'wide', 'leg day')"
    )
    conn.execute(
        "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
        "VALUES ('c1', 'j1', 1, 'a.png', 1, true)"
    )
    conn.commit()

    adapter = MockNotificationAdapter()
    triggered = live_rollback(conn=conn, user_id="u1", baseline_rate=0.90, adapter=adapter)

    assert triggered is False
    assert adapter.sent == []
