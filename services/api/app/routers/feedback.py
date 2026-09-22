"""`POST /feedback/{pair,caption,thumb}` write straight into `image_pairs`/
`caption_pairs`/`thumbs`, each row scoped by `date` (resolved to the day's
`job_id`, which is what those tables actually FK against) rather than
exposing `job_id` as a public identifier a frontend would need to
remember.

`/regenerate` enqueues `app.worker.regenerate_panel_job` (identity-gated
regeneration, reusing the critic's own retry machinery) since it can't run
synchronously in the API process — it calls real image/LLM providers.

`/pair` with `source="ab"` also enqueues `app.worker.apply_pair_choice_job`,
which flips the strip to the tapped image. That job needs no external API
call (the tapped image already exists in `Storage`), so it completes in
roughly the time Pillow takes to recompose a strip.
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any, Literal
from uuid import uuid4

from arq import ArqRedis
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.db import fetch_one, get_conn
from app.deps import arq_pool_dep, current_user_id

router = APIRouter(prefix="/feedback", tags=["feedback"])


def _job_id_for_date(user_id: str, date: date_cls) -> str:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT job_id FROM days WHERE user_id = %s AND date = %s AND kind = 'daily'",
            (user_id, date),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="no day for that date")
    return str(row["job_id"])


class PairIn(BaseModel):
    date: date_cls
    panel_id: int
    chosen: str
    rejected: str
    source: Literal["ab", "regenerate"]


class PickIn(BaseModel):
    """Best-of-3: the person picked `picked` as their favourite of the options shown
    (`others` are the ones they passed on). One tap, learned as the pick beating each other."""

    date: date_cls
    panel_id: int
    picked: str
    others: list[str] = Field(min_length=1, max_length=5)


class RatingIn(BaseModel):
    """0 off, 1 ok, 2 great — for one image (usually the one in the strip)."""

    date: date_cls
    panel_id: int
    url: str
    rating: int = Field(ge=0, le=2)


class CaptionIn(BaseModel):
    date: date_cls
    panel_id: int
    original: str
    edited: str


class ThumbIn(BaseModel):
    date: date_cls
    panel_id: int
    value: Literal[1, -1]


@router.post("/pair", status_code=201)
async def feedback_pair(
    body: PairIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, Any]:
    job_id = _job_id_for_date(user_id, body.date)
    with get_conn() as conn:
        row = fetch_one(
            conn,
            """
            INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """,
            (job_id, body.panel_id, body.chosen, body.rejected, body.source),
        )
    if body.source == "ab":
        # `source == "regenerate"` pairs are written directly by
        # `regenerate_panel_job` itself, never through this route, so this
        # only ever fires for a genuine "Two quick taps" UI pick.
        await arq_pool.enqueue_job(
            "apply_pair_choice_job", job_id, user_id, body.panel_id, body.chosen
        )
    return {"id": row["id"]}


@router.post("/pick", status_code=201)
async def feedback_pick(
    body: PickIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, Any]:
    """A quick tap is a pick of the favourite of a panel's three options. Writes one
    `image_pairs` row per option passed on, all sharing a `tap_id`, so the preference model can
    learn it as a single tap (`app.preference.retrain._load_taps`) while older single-pair rows
    keep working. The strip switches to the pick exactly as `/pair` does."""
    if body.picked in body.others:
        raise HTTPException(status_code=422, detail="the picked image can't also be passed on")
    job_id = _job_id_for_date(user_id, body.date)
    tap_id = uuid4().hex
    with get_conn() as conn:
        for other in dict.fromkeys(body.others):  # keep order, drop repeats
            conn.execute(
                """
                INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, tap_id)
                VALUES (%s, %s, %s, %s, 'ab', %s)
                """,
                (job_id, body.panel_id, body.picked, other, tap_id),
            )
    await arq_pool.enqueue_job("apply_pair_choice_job", job_id, user_id, body.panel_id, body.picked)
    return {"tap_id": tap_id}


@router.post("/rating", status_code=201)
async def feedback_rating(
    body: RatingIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, Any]:
    """A rating (off / ok / great) of a single image. A pick says which of three is best; a
    rating also says whether an image is good at all, so the models learn from both. A changed
    mind is a new row (the latest per image counts). Nothing on screen changes, so only the
    learning is re-run."""
    job_id = _job_id_for_date(user_id, body.date)
    with get_conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM candidates WHERE day_id = %s AND panel_id = %s AND url = %s",
            (job_id, body.panel_id, body.url),
        ).fetchone()
        if exists is None:
            raise HTTPException(status_code=404, detail="no such image for that panel")
        row = fetch_one(
            conn,
            "INSERT INTO image_ratings (day_id, panel_id, url, rating) VALUES (%s, %s, %s, %s) "
            "RETURNING id",
            (job_id, body.panel_id, body.url, body.rating),
        )
    await arq_pool.enqueue_job("rebuild_preference_job", user_id)
    return {"id": row["id"]}


MAX_CAPTION_CHARS = 60  # `Panel.caption_a` in the contract


@router.post("/caption", status_code=201)
async def feedback_caption(
    body: CaptionIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, Any]:
    edited = body.edited.strip()
    if not edited:
        raise HTTPException(status_code=422, detail="the caption can't be empty")
    if len(edited) > MAX_CAPTION_CHARS:
        raise HTTPException(
            status_code=422, detail=f"captions are at most {MAX_CAPTION_CHARS} characters"
        )
    job_id = _job_id_for_date(user_id, body.date)
    with get_conn() as conn:
        row = fetch_one(
            conn,
            """
            INSERT INTO caption_pairs (day_id, panel_id, original, edited)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (job_id, body.panel_id, body.original, edited),
        )
        conn.execute(
            "UPDATE panels SET caption_a = %s WHERE day_id = %s AND panel_id = %s",
            (edited, job_id, body.panel_id),
        )
    # The composed strip (and the file you download) has captions baked in —
    # re-compose it so it matches what's on screen. No image generation.
    await arq_pool.enqueue_job("recompose_strip_job", job_id, user_id)
    return {"id": row["id"], "caption": edited}


@router.post("/thumb", status_code=201)
async def feedback_thumb(body: ThumbIn, user_id: str = Depends(current_user_id)) -> dict[str, Any]:
    job_id = _job_id_for_date(user_id, body.date)
    with get_conn() as conn:
        row = fetch_one(
            conn,
            "INSERT INTO thumbs (day_id, panel_id, value) VALUES (%s, %s, %s) RETURNING id",
            (job_id, body.panel_id, body.value),
        )
    return {"id": row["id"]}


class RegenerateIn(BaseModel):
    # What the person wants changed ("make me look less tired"); goes into the
    # image prompt. Capped so it can't bloat the prompt.
    note: str = Field(default="", max_length=300)


# Separate router (no "/feedback" prefix): this endpoint's real path is
# `/days/{date}/panels/{id}/regenerate`, not under `/feedback`, even
# though it belongs with these feedback-write endpoints logically.
days_router = APIRouter(tags=["feedback"])


@days_router.post("/days/{date}/panels/{panel_id}/regenerate", status_code=202)
async def regenerate_panel(
    date: date_cls,
    panel_id: int,
    body: RegenerateIn,
    user_id: str = Depends(current_user_id),
    arq_pool: ArqRedis = Depends(arq_pool_dep),
) -> dict[str, str]:
    job_id = _job_id_for_date(user_id, date)
    await arq_pool.enqueue_job("regenerate_panel_job", job_id, user_id, panel_id, body.note)
    return {"job_id": job_id, "status": "regenerating"}
