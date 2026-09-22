"""ml/registry.py — mocks `ml.identity.hf_registry.upload_lora` (already
itself mocked-and-live-verified in `test_hf_registry.py`) and uses a
real, isolated Postgres schema for the `checkpoints` mirror rather than
mocking the database too — "the checkpoints table mirrors the registry"
is only really tested by writing to a real table and reading it back.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv

from ml.registry import list_checkpoints, register_checkpoint

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "comiccanvas_test_registry"


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
    monkeypatch.setenv("HF_REPO", "someuser/comiccanvas-models")
    monkeypatch.setenv("HF_TOKEN", "test-token-not-real")


@pytest.fixture(autouse=True)
def _clean_checkpoints(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
        conn.execute("TRUNCATE checkpoints")
        conn.commit()


@pytest.fixture(autouse=True)
def _isolate_registry_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """`ml.registry`'s own `psycopg.connect` calls (unlike `app.db.
    get_conn`) don't read `DB_SCHEMA` themselves — wrap them to always
    scope to the test schema, mirroring the real `SET search_path`
    pattern used everywhere else in this project."""
    real_connect = psycopg.connect

    def scoped_connect(*args: object, **kwargs: object) -> psycopg.Connection[Any]:
        conn = real_connect(*args, **kwargs)  # type: ignore[arg-type]
        conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
        return conn

    monkeypatch.setattr(psycopg, "connect", scoped_connect)


def test_register_checkpoint_uploads_and_mirrors_into_the_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api = MagicMock()
    monkeypatch.setattr("ml.identity.hf_registry.HfApi", lambda token: fake_api)

    artifact = tmp_path / "reward_head.onnx"
    artifact.write_bytes(b"fake-onnx-bytes")

    url = register_checkpoint(
        checkpoint_id="reward_head_test_1",
        kind="reward_model",
        artifact_path=artifact,
        path_in_repo="reward_model/reward_head_test_1.onnx",
        data_snapshot="heldout_week:28pairs",
        metrics={"learned_pairwise_accuracy": 0.43, "beats_champion": False},
    )

    assert url == (
        "https://huggingface.co/someuser/comiccanvas-models/resolve/main/"
        "reward_model/reward_head_test_1.onnx"
    )
    fake_api.upload_file.assert_called_once()

    checkpoints = list_checkpoints("reward_model")
    assert len(checkpoints) == 1
    assert checkpoints[0]["id"] == "reward_head_test_1"
    assert checkpoints[0]["artifact_url"] == url
    assert checkpoints[0]["metrics"]["beats_champion"] is False


def test_register_checkpoint_is_idempotent_on_conflict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api = MagicMock()
    monkeypatch.setattr("ml.identity.hf_registry.HfApi", lambda token: fake_api)
    artifact = tmp_path / "x.onnx"
    artifact.write_bytes(b"v1")

    register_checkpoint(
        checkpoint_id="dup-id",
        kind="reward_model",
        artifact_path=artifact,
        path_in_repo="reward_model/dup.onnx",
        data_snapshot="snap-1",
        metrics={"n": 1},
    )
    register_checkpoint(
        checkpoint_id="dup-id",
        kind="reward_model",
        artifact_path=artifact,
        path_in_repo="reward_model/dup.onnx",
        data_snapshot="snap-2",
        metrics={"n": 2},
    )

    checkpoints = [c for c in list_checkpoints("reward_model") if c["id"] == "dup-id"]
    assert len(checkpoints) == 1
    assert checkpoints[0]["data_snapshot"] == "snap-2"


def test_list_checkpoints_only_returns_the_requested_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_api = MagicMock()
    monkeypatch.setattr("ml.identity.hf_registry.HfApi", lambda token: fake_api)
    artifact = tmp_path / "x.onnx"
    artifact.write_bytes(b"data")

    register_checkpoint(
        checkpoint_id="rm-1",
        kind="reward_model",
        artifact_path=artifact,
        path_in_repo="a.onnx",
        data_snapshot="s",
        metrics={},
    )
    register_checkpoint(
        checkpoint_id="script-dpo-1",
        kind="script_dpo",
        artifact_path=artifact,
        path_in_repo="b.onnx",
        data_snapshot="s",
        metrics={},
    )

    assert {c["id"] for c in list_checkpoints("reward_model")} == {"rm-1"}
    assert {c["id"] for c in list_checkpoints("script_dpo")} == {"script-dpo-1"}
