"""The four scheduler entrypoints — `remind`, `weekly_recap`,
`export_feedback`, `kick_training`. Each takes `now: datetime` as an
explicit parameter rather than calling `datetime.now()` internally, so a
test can construct the exact `datetime` it wants and assert on it without
a time-mocking library.

"User-local" reminder/recap time is treated as UTC — real per-user timezone
handling needs a `timezone` column `settings` doesn't have yet.
`export_feedback` excludes the held-out week: the set of ISO weeks present
in `ml/goldens/pairs/heldout_week.jsonl` is never re-exported as training
data. `kick_training` runs `ml.reward_model.train.main()` (a tiny CPU-only
BT head) on schedule; script DPO / prompt-writer GRPO / diffusion-DPO need
real GPU compute and aren't built, so it logs that honestly rather than
pretending to kick them off.

Live rollback computes a real drop-detection decision from real
`candidates` data and alerts, but "flip the flag back to champion" has no
real target yet: `WRITER`/`PROMPTER`/`REWARD_HEAD` are read from
`os.environ` by the worker process itself, and no cross-process flag store
exists to let the scheduler change that safely.
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

import psycopg
from arq import ArqRedis
from psycopg.rows import DictRow

logger = logging.getLogger("comiccanvas.scheduler")

_DATA_ROOT = Path(__file__).resolve().parents[3] / "ml" / "data"
_HELDOUT_PATH = (
    Path(__file__).resolve().parents[3] / "ml" / "goldens" / "pairs" / "heldout_week.jsonl"
)


class NotificationAdapter(Protocol):
    def send(self, user_id: str, message: str) -> None: ...


class MockNotificationAdapter:
    """Mock notification sender — no real web push/Telegram/email
    integration exists yet; a real adapter is a follow-up once a channel is
    actually chosen per user (`settings.channel`, already a real column)."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    def send(self, user_id: str, message: str) -> None:
        self.sent.append((user_id, message))


def _iso_week(now: datetime) -> str:
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def _iso_week_from_date_str(date_str: str) -> str:
    iso = datetime.fromisoformat(date_str).isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def held_out_weeks(path: Path | None = None) -> set[str]:
    """The set of ISO weeks the frozen `heldout_week.jsonl` covers —
    `export_feedback` must never re-export these as training data. Returns
    an empty set if the golden file doesn't exist yet.

    `path` defaults to the module-level `_HELDOUT_PATH` read here, at call
    time, not as a `path: Path = _HELDOUT_PATH` default argument, which
    Python evaluates once at function-definition time and would ignore a
    test's `monkeypatch.setattr`."""
    path = path if path is not None else _HELDOUT_PATH
    if not path.is_file():
        return set()
    weeks = set()
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            weeks.add(_iso_week_from_date_str(row["date"]))
    return weeks


def due_reminders(now: datetime, reminder_times: dict[str, str]) -> list[str]:
    hhmm = now.strftime("%H:%M")
    return [user_id for user_id, reminder_time in reminder_times.items() if reminder_time == hhmm]


def remind(
    now: datetime, *, conn: psycopg.Connection[DictRow], adapter: NotificationAdapter
) -> list[str]:
    rows = conn.execute(
        "SELECT user_id, reminder_time FROM settings WHERE reminder_time IS NOT NULL"
    ).fetchall()
    reminder_times = {r["user_id"]: r["reminder_time"] for r in rows}
    due = due_reminders(now, reminder_times)
    for user_id in due:
        adapter.send(user_id, "Time for today's comic!")
    return due


def is_weekly_recap_time(now: datetime) -> bool:
    return now.weekday() == 6 and now.hour == 1 and now.minute == 0


def is_export_feedback_time(now: datetime) -> bool:
    return now.weekday() == 6 and now.hour == 1 and now.minute == 30


def is_kick_training_time(now: datetime) -> bool:
    return now.weekday() == 6 and now.hour == 2 and now.minute == 0


async def weekly_recap(
    now: datetime, *, conn: psycopg.Connection[DictRow], arq_pool: ArqRedis
) -> list[str]:
    if not is_weekly_recap_time(now):
        return []
    iso_week = _iso_week(now)
    rows = conn.execute("SELECT DISTINCT user_id FROM days").fetchall()
    user_ids = [r["user_id"] for r in rows]
    for user_id in user_ids:
        job_id = f"{user_id}:weekly:{iso_week}"
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status, events) "
            "VALUES (%s, %s, 'weekly', 'pending', '[]') ON CONFLICT (id) DO NOTHING",
            (job_id, user_id),
        )
        await arq_pool.enqueue_job("run_weekly_job", job_id, user_id, iso_week)
    return user_ids


def export_feedback(now: datetime, *, conn: psycopg.Connection[DictRow]) -> str | None:
    """Excludes the held-out week by the date each row actually belongs to
    (via its `day_id`'s real `days.date`), not by whether the cron happens
    to be running during that week — "since last run" can span the
    held-out week's own boundary once a second real week exists, and a
    boundary-only check would leak held-out rows into a later export."""
    if not is_export_feedback_time(now):
        return None
    iso_week = _iso_week(now)
    excluded_weeks = held_out_weeks()

    last_run = conn.execute(
        "SELECT created_at FROM training_runs WHERE kind = 'export_feedback' "
        "ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    since = last_run["created_at"] if last_run else datetime.min.replace(tzinfo=UTC)

    tables: dict[str, list[dict[str, Any]]] = {}
    for table in ("image_pairs", "caption_pairs", "thumbs"):
        # `table` is one of the three literals in this loop, never
        # request-derived — safe despite the f-string.
        rows = conn.execute(
            f"SELECT t.*, d.date AS day_date FROM {table} t "
            "JOIN days d ON d.job_id = t.day_id "
            "WHERE t.created_at > %s ORDER BY t.created_at",
            (since,),
        ).fetchall()
        kept = [
            {k: v for k, v in dict(r).items() if k != "day_date"}
            for r in rows
            if _iso_week_from_date_str(r["day_date"].isoformat()) not in excluded_weeks
        ]
        tables[table] = kept

    if not any(tables.values()):
        logger.info("export_feedback: nothing to export (held-out week excluded)")
        return None

    out_dir = _DATA_ROOT / iso_week
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(tables, default=str, sort_keys=True)
    (out_dir / "feedback.json").write_text(payload)
    snapshot_hash = hashlib.sha256(payload.encode()).hexdigest()

    conn.execute(
        "INSERT INTO training_runs (kind, config, snapshot_hash) "
        "VALUES ('export_feedback', %s, %s)",
        (json.dumps({"iso_week": iso_week}), snapshot_hash),
    )
    return snapshot_hash


async def kick_training(now: datetime) -> bool:
    """The `REWARD_HEAD` flag is what a real deployment would flip once a
    promoted checkpoint exists — that flip itself is a human/ops decision
    (or the automated rollback), not something this function does on its
    own."""
    if not is_kick_training_time(now):
        return False

    from ml.reward_model.train import main as train_reward_model

    logger.info("kick_training: due — running the real reward-model retrain")
    try:
        train_reward_model()
    except FileNotFoundError:
        logger.warning("kick_training: no heldout_week.jsonl yet — nothing to train on")
        return True
    logger.info(
        "kick_training: reward model retrained; script DPO/prompt-writer GRPO/"
        "diffusion-DPO need real GPU compute and aren't built — not run"
    )
    return True


def first_pick_rate_by_day(
    conn: psycopg.Connection[DictRow], *, user_id: str, days: int
) -> list[float]:
    """One rate per of the last `days` distinct days that have any
    candidates — "first pick" approximated the same way `GET /learning`
    does: the earliest-created candidate per panel being the one marked
    `chosen`, since no explicit attempt-order column exists."""
    rows = conn.execute(
        """
        SELECT d.date, c.panel_id, c.chosen,
               row_number() OVER (
                   PARTITION BY c.day_id, c.panel_id ORDER BY c.created_at
               ) AS attempt_no
        FROM candidates c
        JOIN days d ON d.job_id = c.day_id
        WHERE d.user_id = %s AND d.kind = 'daily'
        ORDER BY d.date DESC
        """,
        (user_id,),
    ).fetchall()

    by_date: dict[Any, list[bool]] = {}
    for row in rows:
        if row["attempt_no"] != 1:
            continue
        by_date.setdefault(row["date"], []).append(bool(row["chosen"]))

    recent_dates = sorted(by_date, reverse=True)[:days]
    return [sum(by_date[d]) / len(by_date[d]) for d in recent_dates if by_date[d]]


def is_rollback_needed(
    recent_rates: list[float], baseline_rate: float, *, drop_threshold_pts: float = 5.0
) -> bool:
    if not recent_rates:
        return False
    avg_recent = sum(recent_rates) / len(recent_rates)
    return (baseline_rate - avg_recent) * 100 >= drop_threshold_pts


def live_rollback(
    *,
    conn: psycopg.Connection[DictRow],
    user_id: str,
    baseline_rate: float,
    adapter: NotificationAdapter,
    days: int = 3,
) -> bool:
    """Real detection and alert. The "flip the flag back to champion" half
    has no real target yet — see this module's own docstring for why."""
    recent_rates = first_pick_rate_by_day(conn, user_id=user_id, days=days)
    if not is_rollback_needed(recent_rates, baseline_rate):
        return False
    avg_recent = sum(recent_rates) / len(recent_rates) if recent_rates else 0.0
    adapter.send(
        user_id,
        f"ALERT: first-pick rate dropped to {avg_recent:.0%} "
        f"(baseline {baseline_rate:.0%}) over the last {len(recent_rates)} day(s) — "
        "rollback recommended (no automated flag flip wired yet, see cron.py)",
    )
    return True
