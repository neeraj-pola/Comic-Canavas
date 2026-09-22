"""`GET /account/export` (background job, zip download) and
`DELETE /account` (cascading delete).

"Signed download" is a real presigned-GET concept once R2 exists; locally
`LocalFileStorage.get_url()` already serves `/media/*` with no auth (its own
documented dev-only behavior), so the export's download link is that same
URL, not a new signing mechanism. `DELETE /account` runs synchronously
(bounded work: a handful of `storage.delete_prefix` calls plus one
cascading DB delete), no background job needed the way export's
zip-building has. `cost_events.user_id` is `ON DELETE SET NULL`, preserving
aggregate cost history rather than deleting those rows — no row after this
still identifies the deleted user.
"""

from __future__ import annotations

from typing import Any

from arq import ArqRedis
from fastapi import APIRouter, Depends
from storage import Storage

from app.db import fetch_one, get_conn
from app.deps import arq_pool_dep, current_user_id, storage_dep

router = APIRouter(prefix="/account", tags=["account"])


@router.get("/export")
async def export_account(
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, str]:
    with get_conn() as conn:
        job_id = fetch_one(
            conn,
            "INSERT INTO jobs (id, user_id, kind, status) "
            "VALUES (gen_random_uuid()::text, %s, 'export', 'pending') RETURNING id",
            (user_id,),
        )["id"]
    await arq_pool.enqueue_job("export_account_job", job_id, user_id)
    return {"job_id": job_id, "status": "pending"}


@router.delete("")
async def delete_account(
    user_id: str = Depends(current_user_id), storage: Storage = Depends(storage_dep)
) -> dict[str, Any]:
    with get_conn() as conn:
        people_rows = conn.execute(
            "SELECT id FROM people WHERE user_id = %s", (user_id,)
        ).fetchall()
        person_ids = [r["id"] for r in people_rows]
        conn.execute("DELETE FROM users WHERE id = %s", (user_id,))

    storage.delete_prefix(f"strips/{user_id}/")
    storage.delete_prefix(f"audio/{user_id}/")
    for person_id in person_ids:
        storage.delete_prefix(f"people/{person_id}/")

    return {"deleted": True, "user_id": user_id}
