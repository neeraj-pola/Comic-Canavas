"""`GET /account/export` (enqueues) and `DELETE /account` (cascading
delete + real storage prefix cleanup)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import psycopg
import pytest
from fastapi.testclient import TestClient
from psycopg.rows import DictRow
from storage import LocalFileStorage

pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalFileStorage:
    import app.deps as deps_module

    storage = LocalFileStorage(root=tmp_path, base_url="http://testserver")
    monkeypatch.setattr(deps_module, "get_storage", lambda: storage)
    return storage


def test_export_account_enqueues_the_worker_job(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    arq_pool: AsyncMock,
) -> None:
    response = client.get("/account/export", headers=auth_headers("export-user"))

    assert response.status_code == 200
    job_id = response.json()["job_id"]
    arq_pool.enqueue_job.assert_awaited_once_with("export_account_job", job_id, "export-user")
    row = conn.execute("SELECT kind, status FROM jobs WHERE id = %s", (job_id,)).fetchone()
    assert row is not None
    assert row["kind"] == "export"
    assert row["status"] == "pending"


def test_delete_account_cascades_and_clears_storage(
    client: TestClient,
    conn: psycopg.Connection[DictRow],
    auth_headers: Callable[[str], dict[str, str]],
    _local_storage: LocalFileStorage,
) -> None:
    user_id = "delete-user"
    headers = auth_headers(user_id)
    client.get("/settings", headers=headers)  # upserts the user row

    person_id = client.post("/cast", json={"name": "Sam"}, headers=headers).json()["id"]
    _local_storage.put_object(f"people/{person_id}/raw/photo.jpg", b"fake photo bytes")
    _local_storage.put_object(f"strips/{user_id}/2026-09-16/strip.png", b"fake strip bytes")

    response = client.delete("/account", headers=headers)

    assert response.status_code == 200
    assert conn.execute("SELECT id FROM users WHERE id = %s", (user_id,)).fetchone() is None
    assert conn.execute("SELECT id FROM people WHERE id = %s", (person_id,)).fetchone() is None
    assert not (_local_storage.root / f"people/{person_id}").exists()
    assert not (_local_storage.root / f"strips/{user_id}").exists()
