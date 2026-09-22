"""The Storage interface.

One contract, two adapters: `LocalFileStorage` for dev (files under
`.data/`, served by the API at `/media/*`) and `R2Storage` for production
(Cloudflare R2). Both are exercised by the same test suite in
`tests/unit/storage/` — R2's tests skip themselves when credentials aren't
configured.

Methods are synchronous by design: generating a presigned URL is local HMAC
computation (no network round trip) for both backends, and `delete_prefix`
is the one call that may block on real network I/O for R2 — callers on the
async path (API, worker) should run it via `asyncio.to_thread`.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Storage(Protocol):
    """Content-addressed-ish object storage, keyed by path-like strings.

    A `key` looks like `people/{id}/raw/{filename}` or
    `audio/{user}/{date}.webm` — see CLAUDE.md §8.4 for the shapes in use.
    """

    def put_presigned(
        self,
        key: str,
        *,
        content_type: str | None = None,
        max_bytes: int | None = None,
        expires_in: int = 900,
    ) -> str:
        """Return a URL the client can PUT bytes to directly.

        In prod (R2Storage) this is a real presigned S3 PUT URL. In dev
        (LocalFileStorage) it's a signed URL back to the API's own upload
        route; this call only mints the token.
        """
        ...

    def get_url(self, key: str) -> str:
        """Return a URL to fetch/read the object at `key`."""
        ...

    def get_object_by_url(self, url: str) -> bytes:
        """Read back bytes previously written via `put_object`, given the
        exact URL that call (or `get_url`) returned.

        `DayState` only ever carries a `Candidate.url` (not the raw bytes or
        the storage key, which differ per provider and per backend), and
        must stay resumable across process restarts, so the composer can't
        rely on bytes still being in memory from whatever node generated
        them. Reversing `get_url`'s own URL shape (rather than adding a
        parallel `key` field to `Candidate`) keeps `url` the single source
        of truth for "where is this object."
        """
        ...

    def put_object(self, key: str, data: bytes, *, content_type: str | None = None) -> str:
        """Write `data` to `key` directly and return `get_url(key)`.

        For server-side writes where there's no client to hand a presigned
        URL to: a generated placeholder image, a real generator's output
        downloaded and re-uploaded to R2, a rendered composite. Not a
        general bytes-proxying path for user uploads — those go through
        `put_presigned`, since the server never proxies bytes in prod.
        """
        ...

    def delete_prefix(self, prefix: str) -> None:
        """Delete every object whose key starts with `prefix`.

        Used by cast revocation and account deletion: both require zero
        objects left under the prefix afterward.
        """
        ...


class StorageError(Exception):
    """Raised for invalid keys/prefixes (e.g. path traversal) or backend failures."""
