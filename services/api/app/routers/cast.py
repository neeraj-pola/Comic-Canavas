"""Cast (recurring characters) — CRUD, photo intake, face processing, and
identity onboarding.

`/process` runs real, free, local face detection + embedding
(`ml.identity.faces`/`embedding`, self-contained, no paid API)
synchronously, since it's bounded work with no background-job infra to poll
otherwise. Look-card generation (the real LLM vision call) is not done here
even though it's conceptually part of "processing" a person's photos:
`ml.identity.lookcard` needs `app.llm`, which only resolves when
`services/worker` is the process's own `app` — a real conflict, since this
service's own `app` package is already loaded. It runs inside the worker's
`train_character_job` instead, right before `design_master` needs it.

`/train` (real master design via `ml.identity.master.design_master`) costs
real money per call and needs a `confirm=true` body — it enqueues a worker
job rather than blocking the request on a paid, multi-second generation
call.
"""

from __future__ import annotations

import hashlib
import hmac
import time
from typing import Any

from arq import ArqRedis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from storage import LocalFileStorage, Storage

from app.config import get_settings
from app.db import fetch_one, get_conn
from app.deps import arq_pool_dep, current_user_id, storage_dep
from ml.identity.embedding import build_identity_model
from ml.identity.faces import process_batch

router = APIRouter(prefix="/cast", tags=["cast"])


class PersonIn(BaseModel):
    name: str
    # `LookCard.gender_term` is a user-provided value, never vision-guessed
    # (guessing risks misgendering and isn't more reliable than asking).
    # Collected here, at person creation, not deferred to training time.
    gender_term: str = ""
    # Age, same reasoning as gender_term — user-provided, never
    # vision-guessed — and becomes "{age}-year-old" on the look card.
    # Optional — "prefer not to say" leaves it unset.
    age: int | None = Field(default=None, ge=1, le=120)


class PersonPatchIn(BaseModel):
    """A person is auto-created (just a name) the moment the setup
    wizard's first screen loads, before there's been any chance to ask
    for a gender — this lets that same screen fill it in right after,
    rather than requiring person-creation itself to block on it. Both
    fields are optional so the wizard can save gender and age
    independently (`COALESCE` below keeps whichever isn't sent)."""

    gender_term: str | None = None
    age: int | None = Field(default=None, ge=1, le=120)


@router.get("")
async def list_cast(user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    with get_conn() as conn:
        people = conn.execute(
            "SELECT id, name, gender_term, age, consent_at, revoked_at, created_at FROM people "
            "WHERE user_id = %s ORDER BY created_at",
            (user_id,),
        ).fetchall()
        identities = {
            r["person_id"]: r
            for r in conn.execute(
                "SELECT person_id, source_photo_count, mean_self_cosine, look_card, "
                "master_path, generation_path, master_candidates FROM identity_models "
                "WHERE person_id = ANY(%s)",
                ([p["id"] for p in people],),
            ).fetchall()
        }
        training = {
            r["id"].removeprefix("train:"): r
            for r in conn.execute(
                "SELECT id, status, error, events, created_at, updated_at FROM jobs "
                "WHERE kind = 'train' AND id = ANY(%s)",
                ([f"train:{p['id']}" for p in people],),
            ).fetchall()
        }

    def _training(person_id: str) -> dict[str, Any] | None:
        job = training.get(person_id)
        if job is None:
            return None
        events = job["events"] or []
        return {
            "status": job["status"],
            "stage": events[-1]["node"] if events else None,
            "error": job["error"],
            "started_at": job["created_at"],
            "updated_at": job["updated_at"],
        }

    return {
        "people": [
            {
                **dict(p),
                "identity": dict(identities[p["id"]]) if p["id"] in identities else None,
                "training": _training(p["id"]),
            }
            for p in people
        ]
    }


@router.post("", status_code=201)
async def create_person(body: PersonIn, user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    with get_conn() as conn:
        row = fetch_one(
            conn,
            "INSERT INTO people (id, user_id, name, gender_term, age) "
            "VALUES (gen_random_uuid()::text, %s, %s, %s, %s) "
            "RETURNING id, name, gender_term, age, created_at",
            (user_id, body.name, body.gender_term, body.age),
        )
    return dict(row)


@router.patch("/{person_id}")
async def update_person(
    person_id: str, body: PersonPatchIn, user_id: str = Depends(current_user_id)
) -> dict[str, Any]:
    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        row = fetch_one(
            conn,
            "UPDATE people SET gender_term = COALESCE(%s, gender_term), "
            "age = COALESCE(%s, age) WHERE id = %s "
            "RETURNING id, name, gender_term, age, created_at",
            (body.gender_term, body.age, person_id),
        )
    return dict(row)


def _own_person(conn: Any, user_id: str, person_id: str) -> None:
    row = conn.execute(
        "SELECT id FROM people WHERE id = %s AND user_id = %s", (person_id, user_id)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="person not found")


class PhotoPresignIn(BaseModel):
    filename: str
    content_type: str = "image/jpeg"


@router.post("/{person_id}/photos/presign")
async def presign_photo(
    person_id: str,
    body: PhotoPresignIn,
    user_id: str = Depends(current_user_id),
    storage: Storage = Depends(storage_dep),
) -> dict[str, Any]:
    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        key = f"people/{person_id}/raw/{body.filename}"
        upload_url = storage.put_presigned(key, content_type=body.content_type)
        row = fetch_one(
            conn,
            "INSERT INTO person_photos (person_id, storage_key, status) "
            "VALUES (%s, %s, 'pending') RETURNING id",
            (person_id, key),
        )
    return {"id": row["id"], "upload_url": upload_url, "key": key}


@router.post("/{person_id}/process")
async def process_person(
    person_id: str,
    user_id: str = Depends(current_user_id),
    storage: Storage = Depends(storage_dep),
) -> dict[str, Any]:
    if not isinstance(storage, LocalFileStorage):
        raise HTTPException(status_code=501, detail="processing needs LocalFileStorage for now")

    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        photos = conn.execute(
            "SELECT id, storage_key FROM person_photos WHERE person_id = %s", (person_id,)
        ).fetchall()
    if not photos:
        raise HTTPException(status_code=422, detail="no photos uploaded yet")

    paths = [storage.read_path(p["storage_key"]) for p in photos]
    results = process_batch(paths)

    with get_conn() as conn:
        by_path = {str(p["storage_key"]): p["id"] for p in photos}
        for result in results:
            key = str(result.path.relative_to(storage.root))
            conn.execute(
                "UPDATE person_photos SET status = %s WHERE id = %s",
                (result.status, by_path[key]),
            )

    usable = [r for r in results if r.status == "usable" and r.embedding is not None]
    if not usable:
        return {"photos": [{"status": r.status} for r in results], "identity": None}

    embeddings = [r.embedding for r in usable if r.embedding is not None]
    model = build_identity_model(person_id, embeddings)

    embedding_literal = "[" + ",".join(str(x) for x in (model.embedding or [])) + "]"
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO identity_models (person_id, embedding, source_photo_count, mean_self_cosine)
            VALUES (%s, %s::vector, %s, %s)
            ON CONFLICT (person_id) DO UPDATE SET
                embedding = EXCLUDED.embedding,
                source_photo_count = EXCLUDED.source_photo_count,
                mean_self_cosine = EXCLUDED.mean_self_cosine,
                updated_at = now()
            """,
            (person_id, embedding_literal, model.source_photo_count, model.mean_self_cosine),
        )

    return {
        "photos": [{"status": r.status} for r in results],
        "identity": {
            "source_photo_count": model.source_photo_count,
            "mean_self_cosine": model.mean_self_cosine,
        },
    }


class TrainIn(BaseModel):
    confirm: bool = False


@router.post("/{person_id}/train", status_code=202)
async def train_person(
    person_id: str,
    body: TrainIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, str]:
    """The worker's `train_character_job` builds the look card (the real
    vision call, needs `app.llm`, worker-only — see this router's module
    docstring) before running `design_master` itself, not here in
    `/process`, since that import only resolves inside the worker
    process."""
    if not body.confirm:
        raise HTTPException(
            status_code=422, detail="pass confirm=true — this spends real money (fal.ai)"
        )
    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        row = conn.execute(
            "SELECT source_photo_count FROM identity_models WHERE person_id = %s", (person_id,)
        ).fetchone()
    if row is None or not row["source_photo_count"]:
        raise HTTPException(status_code=422, detail="run /process first (needs usable photos)")

    # Atomic in-flight guard: the upsert only fires — and only `RETURNING`s
    # a row — when no pending/running training job exists for this person,
    # so a double click (or two tabs) can't enqueue a second paid run.
    job_id = f"train:{person_id}"
    with get_conn() as conn:
        claimed = conn.execute(
            """
            INSERT INTO jobs (id, user_id, kind, status, events)
            VALUES (%s, %s, 'train', 'pending', '[]'::jsonb)
            ON CONFLICT (id) DO UPDATE
                SET status = 'pending', events = '[]'::jsonb, error = NULL, updated_at = now()
                WHERE jobs.status <> ALL(%s)
            RETURNING id
            """,
            (job_id, user_id, ["pending", "running"]),
        ).fetchone()
    if claimed is None:
        return {"person_id": person_id, "status": "training"}

    await arq_pool.enqueue_job("train_character_job", person_id)
    return {"person_id": person_id, "status": "training"}


class MasterPickIn(BaseModel):
    candidate_id: str


@router.post("/{person_id}/master")
async def pick_master(
    person_id: str,
    body: MasterPickIn,
    user_id: str = Depends(current_user_id),
    storage: Storage = Depends(storage_dep),
) -> dict[str, Any]:
    """The person's own choice of master ("which one feels like you") —
    promotes one of the candidates `train_character_job` saved to
    `master.png`, records the same provenance fields an auto-approve would
    have written, and discards the rest. Free: no generation happens here."""
    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        row = conn.execute(
            "SELECT master_candidates FROM identity_models WHERE person_id = %s", (person_id,)
        ).fetchone()
        candidates = (row["master_candidates"] if row else None) or []
        chosen = next((c for c in candidates if c["id"] == body.candidate_id), None)
        if chosen is None:
            raise HTTPException(
                status_code=404, detail="candidate not found — retrain to get new options"
            )
        master_url = storage.put_object(
            f"people/{person_id}/master.png",
            storage.get_object_by_url(chosen["url"]),
            content_type="image/png",
        )
        conn.execute(
            """
            UPDATE identity_models SET
                master_path = %s, master_approved_at = now(), master_candidate_id = %s,
                master_seed = %s, style_card_version = %s, generation_path = 'reference',
                master_candidates = NULL, updated_at = now()
            WHERE person_id = %s
            """,
            (master_url, chosen["id"], chosen["seed"], chosen["style_card_version"], person_id),
        )
    storage.delete_prefix(f"people/{person_id}/master_candidates/")
    return {"person_id": person_id, "master_path": master_url}


@router.delete("/{person_id}")
async def revoke_person(
    person_id: str,
    user_id: str = Depends(current_user_id),
    storage: Storage = Depends(storage_dep),
) -> dict[str, Any]:
    with get_conn() as conn:
        _own_person(conn, user_id, person_id)
        conn.execute("DELETE FROM people WHERE id = %s", (person_id,))
    storage.delete_prefix(f"people/{person_id}/")
    return {"revoked": True, "person_id": person_id}


class InviteIn(BaseModel):
    expires_in: int = 7 * 24 * 3600


@router.post("/invite")
async def create_invite(body: InviteIn, user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    """A signed, stateless token — no `invites` table exists — with
    `expires`+`user_id` embedded and HMAC-signed with the same local secret
    `LocalFileStorage`'s own upload tokens use, verified the same way
    (`hmac.compare_digest`)."""
    expires_at = int(time.time()) + body.expires_in
    secret = get_settings().storage_secret.encode()
    message = f"{user_id}:{expires_at}".encode()
    signature = hmac.new(secret, message, hashlib.sha256).hexdigest()
    token = f"{user_id}:{expires_at}:{signature}"
    return {"token": token, "expires_at": expires_at}


def verify_invite(token: str) -> str:
    """Returns the inviting user's id, or raises. Used by the (future)
    accept-invite endpoint once recurring-character onboarding is built."""
    try:
        user_id, expires_str, signature = token.split(":")
        expires_at = int(expires_str)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="malformed invite token") from exc
    if time.time() > expires_at:
        raise HTTPException(status_code=400, detail="invite token expired")
    secret = get_settings().storage_secret.encode()
    expected = hmac.new(secret, f"{user_id}:{expires_at}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, signature):
        raise HTTPException(status_code=400, detail="invalid invite token")
    return user_id
