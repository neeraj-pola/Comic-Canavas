"""LocalFileStorage and R2Storage must satisfy the same contract.

The R2 tests opt back into real sockets (pytest-socket disables them
globally, see root pyproject.toml) and skip themselves when no live R2
bucket is configured. They start passing for real once real credentials
are put in the environment.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from storage import LocalFileStorage, R2Storage, StorageError

R2_ENV_VARS = ["R2_ENDPOINT", "R2_BUCKET", "R2_ACCESS_KEY", "R2_SECRET_KEY", "R2_PUBLIC_BASE"]


def _r2_creds() -> dict[str, str] | None:
    values = {name: os.environ.get(name) for name in R2_ENV_VARS}
    if not all(values.values()):
        return None
    return {k: v for k, v in values.items() if v is not None}


@pytest.fixture
def local_storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000", secret="test-secret")


@pytest.fixture
def r2_storage() -> R2Storage:
    creds = _r2_creds()
    if creds is None:
        pytest.skip(f"R2 credentials not set ({', '.join(R2_ENV_VARS)})")
    return R2Storage(
        endpoint=creds["R2_ENDPOINT"],
        bucket=creds["R2_BUCKET"],
        access_key=creds["R2_ACCESS_KEY"],
        secret_key=creds["R2_SECRET_KEY"],
        public_base=creds["R2_PUBLIC_BASE"],
    )


# --- shared contract: both backends must satisfy this shape ---------------


def _assert_satisfies_storage_contract(storage: LocalFileStorage | R2Storage, key: str) -> None:
    presigned = storage.put_presigned(key, content_type="image/jpeg")
    assert presigned.startswith("http")

    url = storage.get_url(key)
    assert url.startswith("http")
    assert key in url

    put_url = storage.put_object(key, b"fake object bytes", content_type="image/png")
    assert put_url == url
    assert storage.get_object_by_url(put_url) == b"fake object bytes"

    # A prefix with nothing under it must be a safe no-op, not an error.
    storage.delete_prefix(f"{key}-does-not-exist/")


def test_local_storage_satisfies_contract(local_storage: LocalFileStorage) -> None:
    _assert_satisfies_storage_contract(local_storage, "people/u1/raw/photo.jpg")


@pytest.mark.enable_socket
def test_r2_storage_satisfies_contract(r2_storage: R2Storage) -> None:
    _assert_satisfies_storage_contract(r2_storage, "people/u1/raw/photo.jpg")


# --- LocalFileStorage specifics --------------------------------------------


def test_local_put_object_round_trips(local_storage: LocalFileStorage) -> None:
    key = "strips/u1/2026-09-07/panel-1.png"
    local_storage.put_object(key, b"fake png bytes", content_type="image/png")
    assert local_storage.read_path(key).read_bytes() == b"fake png bytes"


def test_local_round_trip_write_read_delete(local_storage: LocalFileStorage) -> None:
    key = "audio/u1/2026-09-06.webm"
    local_storage.write_bytes(key, b"fake audio bytes")

    assert local_storage.read_path(key).read_bytes() == b"fake audio bytes"

    local_storage.delete_prefix("audio/u1")
    assert not local_storage.read_path(key).exists()


def test_local_upload_token_round_trips(local_storage: LocalFileStorage) -> None:
    url = local_storage.put_presigned("people/u1/raw/a.jpg", expires_in=60)
    query = dict(part.split("=") for part in url.split("?", 1)[1].split("&"))
    from urllib.parse import unquote

    key = unquote(query["key"])
    expires = int(query["expires"])
    signature = query["signature"]

    assert local_storage.verify_upload_token(key, expires, signature)
    assert not local_storage.verify_upload_token(key, expires, "wrong-signature")
    assert not local_storage.verify_upload_token("other/key.jpg", expires, signature)


@pytest.mark.parametrize("bad_key", ["../escape.jpg", "/etc/passwd", "a/../../b.jpg"])
def test_local_storage_rejects_path_traversal(
    local_storage: LocalFileStorage, bad_key: str
) -> None:
    with pytest.raises(StorageError):
        local_storage.get_url(bad_key)


def test_local_get_object_by_url_rejects_a_foreign_url(local_storage: LocalFileStorage) -> None:
    with pytest.raises(StorageError):
        local_storage.get_object_by_url("http://not-this-storage.example/media/x.png")
