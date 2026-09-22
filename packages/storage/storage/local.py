"""Dev storage adapter: files under `.data/`, served by the API at `/media/*`."""

from __future__ import annotations

import hashlib
import hmac
import shutil
import time
from pathlib import Path
from urllib.parse import urlencode

from .base import StorageError


def _safe_join(root: Path, key: str) -> Path:
    """Resolve `key` under `root`, rejecting anything that would escape it."""
    if not key or key.startswith("/") or ".." in Path(key).parts:
        raise StorageError(f"unsafe storage key: {key!r}")
    resolved_root = root.resolve()
    candidate = (resolved_root / key).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise StorageError(f"unsafe storage key: {key!r}")
    return candidate


class LocalFileStorage:
    """Filesystem-backed `Storage` for local dev (no Docker, no R2 creds needed).

    `secret` signs upload tokens so `put_presigned` URLs are scoped to the
    exact key they were minted for — verify with `verify_upload_token` from
    the upload route.
    """

    def __init__(
        self,
        root: str | Path = ".data",
        *,
        base_url: str = "http://localhost:8000",
        secret: str = "dev-local-storage-secret",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self._secret = secret.encode()

    def _sign(self, key: str, expires_at: int) -> str:
        msg = f"{key}:{expires_at}".encode()
        return hmac.new(self._secret, msg, hashlib.sha256).hexdigest()

    def put_presigned(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
        expires_in: int = 900,
    ) -> str:
        _safe_join(self.root, key)  # validate early, before minting a token
        expires_at = int(time.time()) + expires_in
        signature = self._sign(key, expires_at)
        params = {"key": key, "expires": expires_at, "signature": signature}
        return f"{self.base_url}/media/upload?{urlencode(params)}"

    def verify_upload_token(self, key: str, expires: int, signature: str) -> bool:
        if int(time.time()) > expires:
            return False
        expected = self._sign(key, expires)
        return hmac.compare_digest(expected, signature)

    def get_url(self, key: str) -> str:
        _safe_join(self.root, key)
        return f"{self.base_url}/media/{key}"

    def get_object_by_url(self, url: str) -> bytes:
        prefix = f"{self.base_url}/media/"
        if not url.startswith(prefix):
            raise StorageError(f"url not served by this storage: {url!r}")
        key = url[len(prefix) :]
        return self.read_path(key).read_bytes()

    def put_object(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        # content_type unused: LocalFileStorage serves everything through
        # the API's /media/* route, which sets it by file extension.
        self.write_bytes(key, data)
        return self.get_url(key)

    def delete_prefix(self, prefix: str) -> None:
        target = _safe_join(self.root, prefix) if prefix else self.root
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()

    # --- helpers used by the upload/serve routes, not part of Storage ---

    def write_bytes(self, key: str, data: bytes) -> Path:
        path = _safe_join(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def read_path(self, key: str) -> Path:
        return _safe_join(self.root, key)
