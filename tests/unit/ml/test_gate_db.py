"""`ml.evals.gate.record_decision`'s real write onto the real
`training_runs` table — isolated the same way `test_registry.py` is
(`DB_SCHEMA` + a real `alembic upgrade head`).
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv
from psycopg.rows import dict_row

from ml.evals.gate import GateDecision

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "comiccanvas_test_gate"


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


@pytest.fixture
def training_run_id(database_url: str) -> Iterator[int]:
    conn = psycopg.connect(database_url, row_factory=dict_row)
    conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
    conn.execute("TRUNCATE training_runs RESTART IDENTITY")
    row = conn.execute(
        "INSERT INTO training_runs (kind) VALUES ('reward_model') RETURNING id"
    ).fetchone()
    conn.commit()
    assert row is not None
    run_id = int(row["id"])
    yield run_id
    conn.close()


@pytest.fixture(autouse=True)
def _isolate_gate_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    real_connect = psycopg.connect

    def scoped_connect(*args: object, **kwargs: object) -> psycopg.Connection[Any]:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
        return conn

    monkeypatch.setattr(psycopg, "connect", scoped_connect)


def test_record_decision_writes_promote_and_reasons(
    database_url: str, training_run_id: int
) -> None:
    from ml.evals.gate import record_decision

    record_decision(training_run_id, GateDecision(promote=True, reasons=[]))

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
        row = conn.execute(
            "SELECT decision, reasons FROM training_runs WHERE id = %s", (training_run_id,)
        ).fetchone()

    assert row is not None
    assert row["decision"] == "promote"
    assert row["reasons"] == []


def test_record_decision_writes_reject_with_reasons(
    database_url: str, training_run_id: int
) -> None:
    from ml.evals.gate import record_decision

    record_decision(
        training_run_id, GateDecision(promote=False, reasons=["reward accuracy too low"])
    )

    with psycopg.connect(database_url, row_factory=dict_row) as conn:
        conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
        row = conn.execute(
            "SELECT decision, reasons FROM training_runs WHERE id = %s", (training_run_id,)
        ).fetchone()

    assert row is not None
    assert row["decision"] == "reject"
    assert row["reasons"] == ["reward accuracy too low"]
