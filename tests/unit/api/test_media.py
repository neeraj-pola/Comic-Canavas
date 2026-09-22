"""Presigned upload + serve, via a real `LocalFileStorage` pointed at a
tmp dir (monkeypatched onto `app.storage.get_storage`)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from storage import LocalFileStorage

pytestmark = pytest.mark.enable_socket


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> LocalFileStorage:
    import app.deps as deps_module

    storage = LocalFileStorage(root=tmp_path, base_url="http://testserver")
    monkeypatch.setattr(deps_module, "get_storage", lambda: storage)
    return storage


def test_upload_then_serve_round_trips(
    client: TestClient, _local_storage: LocalFileStorage
) -> None:
    upload_url = _local_storage.put_presigned("audio/u1/2026-09-16.webm", content_type="audio/webm")
    path_and_query = upload_url.split("http://testserver", 1)[1]

    upload = client.put(path_and_query, content=b"fake audio bytes")
    assert upload.status_code == 204

    serve = client.get("/media/audio/u1/2026-09-16.webm")
    assert serve.status_code == 200
    assert serve.content == b"fake audio bytes"
    assert serve.headers["content-type"] == "audio/webm"
    # Candidate/strip URLs are not immutable — `seed_for_panel` derives a
    # deterministic seed from `job_id` + `panel_id` alone, so a "Redo
    # today's strip" can overwrite the exact same URL with different
    # content. `no-store` stops a browser from serving a stale cached image.
    assert serve.headers["cache-control"] == "no-store"


def test_upload_rejects_an_expired_or_tampered_token(client: TestClient) -> None:
    response = client.put("/media/upload?key=x&expires=1&signature=bad")
    assert response.status_code == 403


def test_serve_404s_for_a_missing_key(client: TestClient) -> None:
    response = client.get("/media/does/not/exist.png")
    assert response.status_code == 404
