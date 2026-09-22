"""The real Alembic migration (`migrations/versions/af5692fa786e_*`) against
a real local Postgres (`DATABASE_URL`), isolated in its own
`comiccanvas_test_migrations` schema so tests never touch dev data, via
`migrations/env.py`'s `DB_SCHEMA` mechanism.

Lives under `tests/integration/`, not `tests/unit/`: it needs a real database
connection (`enable_socket`, mirroring `test_memory_postgres.py`) and drives
the real Alembic Python API rather than testing a pure function. Runs from
the repo root via plain `pytest` (no `app.*` import, unlike the worker's own
tests) — see root `pyproject.toml`'s `testpaths`.

Skips outright if `DATABASE_URL` isn't configured, mirroring
`test_storage_contract.py`'s R2-credentials skip.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from dotenv import find_dotenv, load_dotenv

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_SCHEMA = "comiccanvas_test_migrations"

# The full table list this migration creates (excludes memory_beats/
# memory_people/memory_places — separate, untouched tables, see the
# migration's own docstring for why they're deliberately not here).
EXPECTED_TABLES = {
    "users",
    "jobs",
    "people",
    "person_photos",
    "identity_models",
    "days",
    "beats",
    "panels",
    "candidates",
    "image_prompts",
    "image_pairs",
    "caption_pairs",
    "thumbs",
    "settings",
    "checkpoints",
    "training_runs",
    "cost_events",
}


def _database_url() -> str | None:
    load_dotenv(find_dotenv(usecwd=True))
    return os.environ.get("DATABASE_URL")


def _alembic_config() -> Config:
    cfg = Config(str(REPO_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    return cfg


def _drop_test_schema(database_url: str) -> None:
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')


def _table_names(database_url: str) -> set[str]:
    with (
        psycopg.connect(database_url) as conn,
        conn.cursor() as cur,
    ):
        cur.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = %s",
            (TEST_SCHEMA,),
        )
        return {row[0] for row in cur.fetchall()}


@pytest.fixture
def database_url() -> Iterator[str]:
    url = _database_url()
    if not url:
        pytest.skip("DATABASE_URL not configured; see the README for local Postgres setup")
    os.environ["DB_SCHEMA"] = TEST_SCHEMA
    _drop_test_schema(url)
    try:
        yield url
    finally:
        _drop_test_schema(url)
        del os.environ["DB_SCHEMA"]


def test_upgrade_head_creates_every_table_in_the_isolated_schema(database_url: str) -> None:
    command.upgrade(_alembic_config(), "head")

    tables = _table_names(database_url)
    assert tables >= EXPECTED_TABLES
    # Guards against Alembic's own bookkeeping table silently falling
    # through search_path to `public` instead of genuinely living in the
    # isolated schema.
    assert "alembic_version" in tables


def test_downgrade_base_removes_exactly_those_tables(database_url: str) -> None:
    command.upgrade(_alembic_config(), "head")
    command.downgrade(_alembic_config(), "base")

    tables = _table_names(database_url)
    assert tables.isdisjoint(EXPECTED_TABLES)


def test_foreign_keys_enforce_the_real_chain(database_url: str) -> None:
    command.upgrade(_alembic_config(), "head")

    with psycopg.connect(database_url, options=f"-csearch_path={TEST_SCHEMA},public") as conn:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO users (id) VALUES ('u1')")
            cur.execute(
                "INSERT INTO jobs (id, user_id, kind, status) VALUES ('j1', 'u1', 'daily', 'done')"
            )
            cur.execute(
                "INSERT INTO days (job_id, user_id, date, source) "
                "VALUES ('j1', 'u1', '2026-09-16', 'text')"
            )
            cur.execute(
                "INSERT INTO beats (day_id, beat_id, time, place, event, emotion, "
                "importance, humor) VALUES ('j1', 'b1', 'evening', 'gym', 'ran', "
                "'content', 0.5, 0.2)"
            )
            cur.execute(
                "INSERT INTO panels (day_id, panel_id, beat_id, place, time_of_day, "
                "expression, action, framing, caption_a) VALUES "
                "('j1', 1, 'b1', 'gym', 'evening', 'content', 'running', 'wide', 'ran today')"
            )
            cur.execute(
                "INSERT INTO image_prompts (id, day_id, panel_id, generator, "
                "character_clause, environment_clause, positive, negative, seed) VALUES "
                "('p1', 'j1', 1, 'mock', 'a person', 'a gym', 'a person, a gym', "
                "'blurry', 4176670031)"
            )
            cur.execute(
                "INSERT INTO candidates (id, day_id, panel_id, url, seed, prompt_id) VALUES "
                "('c1', 'j1', 1, 'http://example.com/c1.png', 42, 'p1')"
            )
        conn.commit()

        # Failure path: an invalid day_id must be rejected, not silently accepted.
        with conn.cursor() as cur, pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO days (job_id, user_id, date, source) "
                "VALUES ('missing-job', 'u1', '2026-09-17', 'text')"
            )
        conn.rollback()

        # An invalid prompt_id must be rejected too.
        with conn.cursor() as cur, pytest.raises(psycopg.errors.ForeignKeyViolation):
            cur.execute(
                "INSERT INTO candidates (id, day_id, panel_id, url, seed, prompt_id) VALUES "
                "('c2', 'j1', 1, 'http://example.com/c2.png', 43, 'missing-prompt')"
            )
        conn.rollback()
