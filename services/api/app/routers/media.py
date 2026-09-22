"""Presigned uploads and serving, the dev-mode counterpart to
`LocalFileStorage.put_presigned`: "the server never proxies bytes" in
production still holds locally in spirit — the API only ever writes
exactly the bytes the client PUTs, scoped to the exact key/expiry a
presigned URL was minted for.

`content_type`/`max_bytes` aren't embedded in the signed token today
(`LocalFileStorage.put_presigned` doesn't fold them into the signature) —
enforced here from the request itself as a best-effort local-dev check,
not a byte-for-byte cryptographic guarantee the way a real presigned S3
URL's policy document would be.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from storage import LocalFileStorage, Storage, StorageError

from app.deps import storage_dep

router = APIRouter(tags=["media"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25MB — generous for a photo/audio memo, not unbounded


@router.put("/media/upload")
async def upload(
    request: Request,
    key: str,
    expires: int,
    signature: str,
    storage: Storage = Depends(storage_dep),
) -> Response:
    if not isinstance(storage, LocalFileStorage):
        raise HTTPException(status_code=400, detail="upload route only serves LocalFileStorage")
    if not storage.verify_upload_token(key, expires, signature):
        raise HTTPException(status_code=403, detail="invalid or expired upload token")

    body = await request.body()
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="upload too large")

    try:
        storage.write_bytes(key, body)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return Response(status_code=204)


_CONTENT_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".webm": "audio/webm",
    ".json": "application/json",
    ".pdf": "application/pdf",
}


@router.get("/media/{key:path}")
async def serve(key: str, storage: Storage = Depends(storage_dep)) -> Response:
    if not isinstance(storage, LocalFileStorage):
        raise HTTPException(status_code=404, detail="not found")
    try:
        path = storage.read_path(key)
    except StorageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.is_file():
        raise HTTPException(status_code=404, detail="not found")
    suffix = path.suffix.lower()
    content_type = _CONTENT_TYPES.get(suffix, "application/octet-stream")
    # Candidate/strip URLs are NOT immutable — `seed_for_panel` derives a
    # panel's seed purely from `job_id` + `panel_id`, so a redo or a
    # killed-and-resumed job can write to the exact same key as an earlier
    # attempt, overwriting it. `no-store` avoids a browser serving a stale
    # cached image after the underlying file changes; everything served by
    # LocalFileStorage is local content the API can re-read in
    # microseconds, so there's no real cost to never caching it.
    headers = {"Cache-Control": "no-store"}
    return Response(content=path.read_bytes(), media_type=content_type, headers=headers)
