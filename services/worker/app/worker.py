"""arq job entrypoints for the daily and weekly comic pipeline (`arq app.worker.WorkerSettings`,
see Procfile.dev)."""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import zipfile
from collections.abc import Awaitable, Callable
from datetime import date as date_cls
from typing import Any, ClassVar

import numpy as np
import psycopg
from arq.connections import RedisSettings
from psycopg.rows import dict_row
from psycopg.types.json import Json
from storage import LocalFileStorage, Storage

from app.compose import compose_day
from app.config import get_settings
from app.graph import open_day_graph
from app.images.base import IdentityRef
from app.memory import PostgresMemoryStore, get_memory_store, set_memory_store
from app.nodes.critic import choose_best_for_panel
from app.nodes.generate import generate_one_panel
from app.nodes.prompts import write_single_prompt
from app.nodes.score import judge_candidates, score_one_candidate
from app.nodes.script import DEFAULT_HUMOR_LEVEL
from app.persist import (
    append_job_event,
    mark_job_status,
    persist_candidates,
    persist_day_state,
    persist_image_prompts,
)
from app.preference.notes import maybe_refresh_notes
from app.preference.retrain import rebuild_preference
from app.preference.state import load_preference_state
from app.prompts.style_card import load_style_card
from app.weekly_highlights import NoPanelsInWeekError, SourcePanel, choose_highlights
from contracts import Beat, Candidate, DayState, ImagePrompt, LookCard
from ml.identity.faces import process_batch
from ml.identity.lookcard import build_look_card, select_best_crops
from ml.identity.master import MasterRequest, design_master

logger = logging.getLogger("comiccanvas.worker")


class NoApprovedCharacterError(RuntimeError):
    """Raised when a user has no approved character to generate a day for."""

    def __init__(self, user_id: str) -> None:
        super().__init__(f"no approved character found for user {user_id!r}")


def _storage() -> Storage:
    """The active storage backend."""
    settings = get_settings()
    if settings.storage == "r2":
        raise NotImplementedError("R2Storage is not wired up yet")
    return LocalFileStorage(root=settings.local_storage_dir)


def _connect(*, row_factory: Any = None) -> psycopg.Connection[Any]:
    """A Postgres connection, scoped to `DB_SCHEMA` when set (test isolation)."""
    settings = get_settings()
    kwargs: dict[str, Any] = {"row_factory": row_factory} if row_factory else {}
    conn = psycopg.connect(settings.database_url, **kwargs)
    schema = os.environ.get("DB_SCHEMA")
    if schema:
        conn.execute(f'SET search_path TO "{schema}", public')
    return conn


def _load_identity(conn: psycopg.Connection[Any], user_id: str) -> tuple[IdentityRef, LookCard]:
    """The user's approved character reference and look card, or raises if none exists."""
    row = conn.execute(
        """
        SELECT im.master_path, im.look_card
        FROM identity_models im
        JOIN people p ON p.id = im.person_id
        WHERE p.user_id = %s AND im.master_path IS NOT NULL AND im.look_card IS NOT NULL
        ORDER BY p.created_at
        LIMIT 1
        """,
        (user_id,),
    ).fetchone()
    if row is None:
        raise NoApprovedCharacterError(user_id)
    # `master_path` is already a full URL, not a bare storage key.
    identity = IdentityRef(master_url=row["master_path"], trigger_token="")
    look_card = LookCard.model_validate(row["look_card"])
    return identity, look_card


def _load_live_candidates(conn: psycopg.Connection[Any], day_id: str) -> list[Candidate]:
    """A day's candidates straight from the database — the current source of truth for what is
    chosen, since the LangGraph checkpoint can be stale for panels a job isn't touching."""
    rows = conn.execute(
        """
        SELECT id, panel_id, url, seed, scores, face_box, chosen, rejected_reason
        FROM candidates WHERE day_id = %s
        """,
        (day_id,),
    ).fetchall()
    return [
        Candidate(
            id=row["id"],
            panel_id=row["panel_id"],
            url=row["url"],
            seed=row["seed"],
            scores=row["scores"],
            face_box=tuple(row["face_box"]) if row["face_box"] is not None else None,
            chosen=row["chosen"],
            rejected_reason=row["rejected_reason"],
        )
        for row in rows
    ]


async def run_daily_job(
    ctx: dict[str, Any],
    job_id: str,
    user_id: str,
    date_iso: str,
    *,
    text: str | None = None,
    audio_url: str | None = None,
) -> None:
    """Runs the full daily pipeline for one entry and persists the result. Re-running an
    existing `job_id` is a redo: it resets the checkpoint and replaces the previous images,
    picks and ratings for that day."""
    settings = get_settings()
    source = "text" if text is not None else "audio"
    state = DayState(
        job_id=job_id,
        user_id=user_id,
        date=date_cls.fromisoformat(date_iso),
        source=source,
        text=text,
        audio_url=audio_url,
    )

    with _connect(row_factory=dict_row) as setup_conn:
        try:
            identity, look_card = _load_identity(setup_conn, user_id)
        except NoApprovedCharacterError as exc:
            mark_job_status(setup_conn, job_id, status="failed", error=str(exc))
            setup_conn.commit()
            logger.warning("run_daily_job %s: %s", job_id, exc)
            return
        humor_row = setup_conn.execute(
            "SELECT humor FROM settings WHERE user_id = %s", (user_id,)
        ).fetchone()
        humor_level = humor_row["humor"] if humor_row is not None else DEFAULT_HUMOR_LEVEL
        preference = load_preference_state(setup_conn, user_id)
        is_redo = (
            setup_conn.execute("SELECT 1 FROM days WHERE job_id = %s", (job_id,)).fetchone()
            is not None
        )
        mark_job_status(setup_conn, job_id, status="running")
        setup_conn.commit()

    storage = _storage()
    try:
        async with open_day_graph(
            storage=storage,
            identity=identity,
            look_card=look_card,
            memory=get_memory_store(),
            humor_level=humor_level,
            taste_notes=preference.notes if preference else None,
            preference=preference,
            dsn=settings.database_url,
        ) as day_graph:
            if is_redo:
                await day_graph.reset(job_id)
            final_state = state
            async for node_name, current in day_graph.run_streaming(state):
                final_state = current
                with _connect() as event_conn:
                    append_job_event(event_conn, job_id, {"node": node_name})
                    event_conn.commit()
    except Exception as exc:
        with _connect() as fail_conn:
            mark_job_status(fail_conn, job_id, status="failed", error=str(exc))
            fail_conn.commit()
        logger.exception("run_daily_job %s failed", job_id)
        return

    story_url = storage.get_url(f"strips/{user_id}/{state.date.isoformat()}/story.png")
    with _connect() as done_conn:
        if is_redo:
            for table in ("image_ratings", "image_pairs", "candidates"):
                done_conn.execute(f"DELETE FROM {table} WHERE day_id = %s", (job_id,))
        persist_day_state(done_conn, final_state, story_url=story_url)
        mark_job_status(done_conn, job_id, status="done")
        done_conn.commit()
    if is_redo:
        await _learn_from_feedback(user_id, job_id)


def _apply_regeneration(prompt: ImagePrompt, note: str, seed_offset: int) -> ImagePrompt:
    """Folds the person's note into the prompt and gives it a fresh, unused seed."""
    positive = prompt.positive
    if note.strip():
        positive = (
            f"{positive.rstrip('. ')}. Important change requested by the person: {note.strip()}."
        )
    return prompt.model_copy(
        update={"positive": positive, "seed": (prompt.seed or 0) + seed_offset}
    )


async def regenerate_panel_job(
    ctx: dict[str, Any], job_id: str, user_id: str, panel_id: int, note: str
) -> None:
    """Redraws one panel from a saved day, honoring the person's note, and recomposes the strip."""
    settings = get_settings()
    storage = _storage()

    with _connect(row_factory=dict_row) as conn:
        try:
            identity, look_card = _load_identity(conn, user_id)
        except NoApprovedCharacterError as exc:
            logger.warning("regenerate_panel_job %s: %s", job_id, exc)
            return
        regen_preference = load_preference_state(conn, user_id)
    regen_notes = regen_preference.notes if regen_preference else None

    async with open_day_graph(
        storage=storage,
        identity=identity,
        look_card=look_card,
        memory=get_memory_store(),
        dsn=settings.database_url,
    ) as day_graph:
        state = await day_graph.get_checkpointed_state(job_id)
        if state is None or state.script is None:
            logger.warning("regenerate_panel_job %s: no checkpointed state to resume", job_id)
            return
        panel = next((p for p in state.script.panels if p.id == panel_id), None)
        if panel is None:
            logger.warning("regenerate_panel_job %s: no panel %s in script", job_id, panel_id)
            return
        with _connect(row_factory=dict_row) as count_conn:
            prior = count_conn.execute(
                "SELECT count(*) AS n FROM image_pairs "
                "WHERE day_id = %s AND panel_id = %s AND source = 'regenerate'",
                (job_id, panel_id),
            ).fetchone()
        press = (prior["n"] if prior else 0) + 1
        with _connect(row_factory=dict_row) as live_conn:
            live_now = _load_live_candidates(live_conn, job_id)
        previous_chosen = next((c for c in live_now if c.panel_id == panel_id and c.chosen), None)
        attempts = 0
        regenerated_prompts: list[ImagePrompt] = []

        async def regenerate(pid: int) -> list[Candidate]:
            """Writes a fresh prompt for this panel, generates and scores a new batch."""
            prompt, _usd = await write_single_prompt(
                state,
                pid,
                memory=get_memory_store(),
                look_card=look_card,
                taste_notes=regen_notes,
            )
            nonlocal attempts
            prompt = _apply_regeneration(prompt, note, seed_offset=10_000 * press + 100 * attempts)
            attempts += 1
            regenerated_prompts.append(prompt)
            fresh = await generate_one_panel(prompt, storage=storage, identity=identity)
            rescored = list(
                await asyncio.gather(
                    *(
                        score_one_candidate(c, panel, storage=storage, look_card=look_card)
                        for c in fresh
                    )
                )
            )
            rated, _judge_usd = await judge_candidates(
                panel,
                rescored,
                storage=storage,
                look_card=look_card,
                reference_url=identity.master_url,
            )
            return rated

        fresh_candidates = await regenerate(panel_id)
        final_batch, _errors = await choose_best_for_panel(
            panel_id, fresh_candidates, regenerate=regenerate, framing=panel.framing
        )
        new_chosen = next(c for c in final_batch if c.chosen)

    with _connect() as write_conn:
        write_conn.execute(
            "UPDATE candidates SET chosen = false WHERE day_id = %s AND panel_id = %s",
            (job_id, panel_id),
        )
        prompt_by_panel = persist_image_prompts(write_conn, job_id, regenerated_prompts)
        persist_candidates(write_conn, job_id, final_batch, prompt_by_panel=prompt_by_panel)
        if previous_chosen is not None:
            write_conn.execute(
                """
                INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, versions)
                VALUES (%s, %s, %s, %s, 'regenerate', %s)
                """,
                (
                    job_id,
                    panel_id,
                    new_chosen.url,
                    previous_chosen.url,
                    Json({"note": note}),
                ),
            )
        write_conn.commit()

    await _recompose_day(job_id, user_id)


def _overlay_saved_captions(
    conn: psycopg.Connection[Any], job_id: str, state: DayState
) -> DayState:
    """Replaces each panel's caption with the person's saved edit, if any."""
    if state.script is None:
        return state
    saved = {
        r["panel_id"]: r["caption_a"]
        for r in conn.execute(
            "SELECT panel_id, caption_a FROM panels WHERE day_id = %s", (job_id,)
        ).fetchall()
    }
    panels = [
        p.model_copy(update={"caption_a": saved[p.id]}) if p.id in saved else p
        for p in state.script.panels
    ]
    return state.model_copy(update={"script": state.script.model_copy(update={"panels": panels})})


async def _recompose_day(
    job_id: str, user_id: str, *, choose: tuple[int, str] | None = None
) -> bool:
    """Rebuilds a day's composed strip from what is currently in the database — no generation,
    only layout. `choose=(panel_id, url)` first makes that candidate the panel's pick."""
    settings = get_settings()
    storage = _storage()

    with _connect(row_factory=dict_row) as conn:
        try:
            identity, look_card = _load_identity(conn, user_id)
        except NoApprovedCharacterError as exc:
            logger.warning("recompose %s: %s", job_id, exc)
            return False
        live_candidates = _load_live_candidates(conn, job_id)

    if choose is not None:
        panel_id, chosen_url = choose
        if not any(c.panel_id == panel_id and c.url == chosen_url for c in live_candidates):
            logger.warning(
                "recompose %s: url %r not found for panel %s", job_id, chosen_url, panel_id
            )
            return False
        live_candidates = [
            c.model_copy(update={"chosen": c.url == chosen_url}) if c.panel_id == panel_id else c
            for c in live_candidates
        ]

    async with open_day_graph(
        storage=storage,
        identity=identity,
        look_card=look_card,
        memory=get_memory_store(),
        dsn=settings.database_url,
    ) as day_graph:
        state = await day_graph.get_checkpointed_state(job_id)
        if state is None or state.script is None:
            logger.warning("recompose %s: no checkpointed state to resume", job_id)
            return False
        state = state.model_copy(update={"candidates": live_candidates})
        with _connect(row_factory=dict_row) as conn:
            state = _overlay_saved_captions(conn, job_id, state)
        state = compose_day(state, storage=storage)

    story_url = storage.get_url(f"strips/{user_id}/{state.date.isoformat()}/story.png")
    with _connect() as write_conn:
        if choose is not None:
            persist_candidates(
                write_conn, job_id, [c for c in state.candidates if c.panel_id == choose[0]]
            )
        persist_day_state(write_conn, state, story_url=story_url)
        write_conn.commit()
    return True


async def recompose_strip_job(ctx: dict[str, Any], job_id: str, user_id: str) -> None:
    """Rebuilds a day's strip after a caption edit."""
    await _recompose_day(job_id, user_id)


async def apply_pair_choice_job(
    ctx: dict[str, Any], job_id: str, user_id: str, panel_id: int, chosen_url: str
) -> None:
    """Applies an A/B pick: makes it the panel's chosen candidate, recomposes the strip, and
    relearns from it."""
    if not await _recompose_day(job_id, user_id, choose=(panel_id, chosen_url)):
        return

    await _learn_from_feedback(user_id, job_id)


async def rebuild_preference_job(ctx: dict[str, Any], user_id: str) -> None:
    """Relearns from a rating without touching the composed strip."""
    await _learn_from_feedback(user_id, f"rating:{user_id}")


async def _learn_from_feedback(user_id: str, job_id: str) -> None:
    """Rebuilds the preference model from every pick and rating, then refreshes taste notes if
    due. Failures here never affect the feedback that triggered it."""
    storage = _storage()
    try:
        with _connect(row_factory=dict_row) as learn_conn:
            summary = rebuild_preference(learn_conn, user_id, storage)
            learn_conn.commit()
        logger.info("learning %s: preference model %s", job_id, summary)
    except Exception:
        logger.exception("learning %s: preference update failed", job_id)
        return

    try:
        with _connect(row_factory=dict_row) as notes_conn:
            wrote = await maybe_refresh_notes(notes_conn, user_id)
            notes_conn.commit()
        if wrote:
            logger.info("learning %s: refreshed taste notes", job_id)
    except Exception:
        logger.exception("learning %s: taste notes refresh failed", job_id)


async def export_account_job(ctx: dict[str, Any], job_id: str, user_id: str) -> None:
    """Builds a zip of the user's rows, photos and strips and records its download URL."""
    storage = _storage()

    with _connect(row_factory=dict_row) as conn:
        mark_job_status(conn, job_id, status="running")
        conn.commit()
        people = conn.execute("SELECT * FROM people WHERE user_id = %s", (user_id,)).fetchall()
        days = conn.execute("SELECT * FROM days WHERE user_id = %s", (user_id,)).fetchall()
        day_ids = [d["job_id"] for d in days]
        beats = (
            conn.execute("SELECT * FROM beats WHERE day_id = ANY(%s)", (day_ids,)).fetchall()
            if day_ids
            else []
        )
        panels = (
            conn.execute("SELECT * FROM panels WHERE day_id = ANY(%s)", (day_ids,)).fetchall()
            if day_ids
            else []
        )

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "data.json",
            json.dumps(
                {
                    "people": [dict(p) for p in people],
                    "days": [dict(d) for d in days],
                    "beats": [dict(b) for b in beats],
                    "panels": [dict(p) for p in panels],
                },
                default=str,
                indent=2,
            ),
        )
        if isinstance(storage, LocalFileStorage):
            for person in people:
                person_dir = storage.root / f"people/{person['id']}"
                if person_dir.is_dir():
                    for path in person_dir.rglob("*"):
                        if path.is_file():
                            zf.write(path, arcname=f"photos/{path.relative_to(storage.root)}")
            for day in days:
                if not day["strip_url"]:
                    continue
                try:
                    data = storage.get_object_by_url(day["strip_url"])
                except Exception:
                    continue
                zf.writestr(f"strips/{day['date'].isoformat()}/strip.png", data)

    download_url = storage.put_object(
        f"exports/{user_id}/{job_id}.zip", buffer.getvalue(), content_type="application/zip"
    )
    with _connect() as conn:
        append_job_event(conn, job_id, {"download_url": download_url})
        mark_job_status(conn, job_id, status="done")
        conn.commit()


def train_job_id(person_id: str) -> str:
    """The `jobs` row id used for one person's training run."""
    return f"train:{person_id}"


def _train_stage(person_id: str, stage: str) -> None:
    """Appends a training-progress event so `GET /cast` can show it."""
    with _connect() as conn:
        conn.execute(
            "UPDATE jobs SET status = 'running', events = events || %s::jsonb, "
            "updated_at = now() WHERE id = %s",
            (Json([{"node": stage}]), train_job_id(person_id)),
        )
        conn.commit()


def _train_finish(person_id: str, status: str, error: str | None = None) -> None:
    """Marks a training job's terminal status."""
    with _connect() as conn:
        mark_job_status(conn, train_job_id(person_id), status=status, error=error)
        conn.commit()


async def train_character_job(ctx: dict[str, Any], person_id: str) -> None:
    """Runs character training and guarantees the job row ends `done` or `failed`."""
    try:
        await _run_training(person_id)
    except Exception as exc:
        logger.exception("train_character_job %s failed", person_id)
        _train_finish(person_id, "failed", str(exc)[:500] or type(exc).__name__)
        return
    _train_finish(person_id, "done")


async def _run_training(person_id: str) -> None:
    """Builds a look card, generates candidate master images, and saves the top-ranked ones for
    the person to pick from."""
    storage = _storage()
    if not isinstance(storage, LocalFileStorage):
        raise NotImplementedError("train_character_job needs LocalFileStorage for now")

    with _connect(row_factory=dict_row) as conn:
        identity_row = conn.execute(
            "SELECT embedding::text AS embedding, look_card FROM identity_models "
            "WHERE person_id = %s",
            (person_id,),
        ).fetchone()
        photo_rows = conn.execute(
            "SELECT storage_key FROM person_photos WHERE person_id = %s AND status = 'usable'",
            (person_id,),
        ).fetchall()
        person_row = conn.execute(
            "SELECT gender_term, age FROM people WHERE id = %s", (person_id,)
        ).fetchone()

    if identity_row is None or not photo_rows:
        raise RuntimeError("no processed photos found — upload photos first")

    reference_embedding = None
    if identity_row["embedding"]:
        reference_embedding = np.array(
            [float(x) for x in identity_row["embedding"].strip("[]").split(",")]
        )

    _train_stage(person_id, "photos")
    crop_paths = [storage.read_path(r["storage_key"]) for r in photo_rows]
    faces = process_batch(crop_paths)
    usable = [f for f in faces if f.status == "usable"]
    if not usable:
        raise RuntimeError("no usable face crops found in your photos")
    crop_jpegs = select_best_crops(usable)

    _train_stage(person_id, "look_card")
    gender_term = person_row["gender_term"] if person_row is not None else ""
    age = person_row["age"] if person_row is not None else None
    age_descriptor = f"{age}-year-old" if age else ""
    if identity_row["look_card"] is not None:
        look_card = LookCard.model_validate(identity_row["look_card"])
        look_card = look_card.model_copy(
            update={
                "gender_term": gender_term or look_card.gender_term,
                "age_descriptor": age_descriptor or look_card.age_descriptor,
            }
        )
    else:
        look_card = await build_look_card(
            crop_jpegs, gender_term=gender_term, age_descriptor=age_descriptor
        )
    with _connect() as conn:
        conn.execute(
            "UPDATE identity_models SET look_card = %s, updated_at = now() WHERE person_id = %s",
            (Json(look_card.model_dump()), person_id),
        )
        conn.commit()

    request = MasterRequest(
        person_id=person_id,
        crops=crop_jpegs,
        look_card=look_card,
        reference_embedding=reference_embedding,
    )
    _train_stage(person_id, "candidates")
    top, _all_ranked, images_by_id = await design_master(request, confirm=True)
    _train_stage(person_id, "ranking")
    if not top:
        raise RuntimeError("no character candidates survived ranking — try again")

    style = load_style_card()
    saved = []
    for candidate in top:
        url = storage.put_object(
            f"people/{person_id}/master_candidates/{candidate.id}.png",
            images_by_id[candidate.id],
            content_type="image/png",
        )
        saved.append(
            {
                "id": candidate.id,
                "url": url,
                "seed": candidate.seed,
                "rank_score": candidate.rank_score,
                "style_score": candidate.style_score,
                "checklist_score": candidate.checklist.score,
                "style_card_version": style.version,
            }
        )
    with _connect() as conn:
        conn.execute(
            "UPDATE identity_models SET master_candidates = %s, updated_at = now() "
            "WHERE person_id = %s",
            (Json(saved), person_id),
        )
        conn.commit()
    _train_stage(person_id, "review")


def _beat_from_row(row: dict[str, Any]) -> Beat:
    """A `Beat` from a `beats` table row."""
    return Beat(
        id=row["beat_id"],
        time=row["time"],
        place=row["place"],
        place_detail=row["place_detail"],
        event=row["event"],
        emotion=row["emotion"],
        people=row["people"],
        objects=row["objects"],
        importance=row["importance"],
        humor=row["humor"],
        quote=row["quote"],
    )


async def run_weekly_job(ctx: dict[str, Any], job_id: str, user_id: str, iso_week: str) -> None:
    """Builds the weekly recap by choosing highlights from the week's existing daily panels and
    laying them out in the six-panel grid. Draws nothing new."""
    year_str, week_str = iso_week.split("-W")
    monday = date_cls.fromisocalendar(int(year_str), int(week_str), 1)
    sunday = date_cls.fromisocalendar(int(year_str), int(week_str), 7)

    with _connect(row_factory=dict_row) as conn:
        mark_job_status(conn, job_id, status="running")
        conn.commit()
        rows = conn.execute(
            """
            SELECT d.date, d.mood, p.panel_id, p.caption_a, p.action, p.place, p.time_of_day,
                   p.expression, p.framing, c.url
            FROM days d
            JOIN panels p ON p.day_id = d.job_id
            JOIN candidates c ON c.day_id = d.job_id AND c.panel_id = p.panel_id AND c.chosen
            WHERE d.user_id = %s AND d.kind = 'daily' AND d.date BETWEEN %s AND %s
            ORDER BY d.date, p.panel_id
            """,
            (user_id, monday, sunday),
        ).fetchall()

    source = [
        SourcePanel(
            date=r["date"].isoformat(),
            day_mood=r["mood"] or "",
            panel_id=r["panel_id"],
            caption=r["caption_a"],
            action=r["action"],
            place=r["place"],
            time_of_day=r["time_of_day"],
            expression=r["expression"],
            framing=r["framing"],
            url=r["url"],
        )
        for r in rows
    ]
    try:
        highlights, mood, usd = await choose_highlights(source)
    except NoPanelsInWeekError as exc:
        with _connect() as conn:
            mark_job_status(conn, job_id, status="failed", error=str(exc))
            conn.commit()
        return
    except Exception as exc:
        with _connect() as conn:
            mark_job_status(conn, job_id, status="failed", error=str(exc))
            conn.commit()
        logger.exception("run_weekly_job %s failed", job_id)
        return

    try:
        with _connect() as conn:
            conn.execute(
                """
                INSERT INTO days (job_id, user_id, date, source, mood, quiet_day, cost_usd, kind)
                VALUES (%s, %s, %s, 'text', %s, false, %s, 'weekly')
                ON CONFLICT (job_id) DO UPDATE SET
                    mood = EXCLUDED.mood, cost_usd = EXCLUDED.cost_usd, date = EXCLUDED.date
                """,
                (job_id, user_id, sunday, mood, usd),
            )
            conn.execute("DELETE FROM candidates WHERE day_id = %s", (job_id,))
            conn.execute("DELETE FROM panels WHERE day_id = %s", (job_id,))
            for number, (panel, caption) in enumerate(highlights, start=1):
                conn.execute(
                    """
                    INSERT INTO panels (day_id, panel_id, place, time_of_day, expression,
                                        action, framing, caption_a)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        job_id,
                        number,
                        panel.place,
                        panel.time_of_day,
                        panel.expression,
                        panel.action,
                        panel.framing,
                        caption,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen)
                    VALUES (%s, %s, %s, %s, 0, true)
                    """,
                    (f"{job_id}:{number}", job_id, number, panel.url),
                )
            mark_job_status(conn, job_id, status="done")
            conn.commit()
    except Exception as exc:
        with _connect() as conn:
            mark_job_status(conn, job_id, status="failed", error=str(exc))
            conn.commit()
        logger.exception("run_weekly_job %s failed while persisting", job_id)


async def healthcheck(ctx: dict[str, Any]) -> str:
    """arq liveness check."""
    logger.info("worker healthcheck ok")
    return "ok"


async def on_startup(ctx: dict[str, Any]) -> None:
    """Configures logging and the shared memory store for the worker process."""
    logging.basicConfig(level=get_settings().log_level)
    set_memory_store(PostgresMemoryStore(schema="public"))
    logger.info("worker starting up")


class WorkerSettings:
    """arq worker configuration: registered jobs, Redis connection, and timeouts."""

    functions: ClassVar[list[Callable[..., Awaitable[Any]]]] = [
        healthcheck,
        run_daily_job,
        regenerate_panel_job,
        apply_pair_choice_job,
        rebuild_preference_job,
        recompose_strip_job,
        export_account_job,
        train_character_job,
        run_weekly_job,
    ]
    on_startup = staticmethod(on_startup)
    redis_settings = RedisSettings.from_dsn(str(get_settings().redis_url))
    # Generous: a daily job can generate and score a dozen images with real API latency.
    job_timeout = 1800
