"""Shared fixtures for `services/api` router tests. Real Postgres
(`DATABASE_URL`), isolated via `DB_SCHEMA` + a real `alembic upgrade
head` against `comiccanvas_test_api` (`app.db.get_conn`'s own per-call
env read).

`AUTH_MODE=mock` is exercised for real (not bypassed via dependency
override) — `Authorization: Bearer <user_id>` is genuinely verified by
`app.auth`/`app.deps.current_user_id`, matching this project's
"mock the boundary, not the internal logic" convention. The arq pool
dependency IS overridden: enqueueing onto a real redis is fine, but
nothing in these tests runs a live worker to consume it, so job-creating
endpoints assert on the enqueue call itself, not on a job actually
finishing.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from unittest.mock import AsyncMock

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv
from fastapi.testclient import TestClient
from psycopg.rows import DictRow, dict_row

# `enable_socket` (pytest-socket, for the real Postgres TCP connection)
# is a per-test-FILE `pytestmark`, not something a shared conftest.py can
# apply on tests' behalf — every test file using these fixtures declares
# `pytestmark = pytest.mark.enable_socket` itself (matches the existing
# convention in test_migrations.py/test_cron.py/test_worker_jobs.py).

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "comiccanvas_test_api"


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


@pytest.fixture
def conn(database_url: str) -> Iterator[psycopg.Connection[DictRow]]:
    connection = psycopg.connect(database_url, row_factory=dict_row)
    connection.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
    # Order matters: children before parents that aren't already CASCADEd.
    connection.execute("TRUNCATE users CASCADE")
    connection.commit()
    try:
        yield connection
        connection.commit()
    finally:
        connection.close()


@pytest.fixture
def arq_pool() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def client(conn: psycopg.Connection[DictRow], arq_pool: AsyncMock) -> Iterator[TestClient]:
    from app.deps import arq_pool_dep
    from app.main import app

    app.dependency_overrides[arq_pool_dep] = lambda: arq_pool
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


@pytest.fixture
def auth_headers() -> Callable[[str], dict[str, str]]:
    """Factory fixture (pytest has no way to parametrize a plain fixture
    per call) — `auth_headers("some-user")` returns the mock-mode bearer
    header `app.auth`'s real (not overridden) verification expects."""
    return lambda user_id: {"Authorization": f"Bearer {user_id}"}
