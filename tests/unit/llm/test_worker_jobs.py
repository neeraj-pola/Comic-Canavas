"""`app.worker.run_daily_job` — the real end-to-end path from an enqueued
job to persisted relational rows (`app.persist`), run directly (no
arq/redis involved — the same "call the function, not the transport"
convention `test_graph.py` already uses for LangGraph nodes).

Real Postgres (`DATABASE_URL`), isolated via `DB_SCHEMA` + a real
`alembic upgrade head` against `comiccanvas_test_worker_jobs` — includes
the real relational tables `_load_identity`/`persist_day_state` read and
write. The LangGraph checkpointer's own `checkpoints` table would
otherwise collide with the app's own `checkpoints` table (the model
registry) if both lived in the same schema, so `open_day_graph`'s
checkpointer defaults to its own `langgraph` schema, verified here
implicitly by this test actually reaching `status='done'`.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from datetime import date
from pathlib import Path
from uuid import uuid4

import numpy as np
import psycopg
import pytest
from dotenv import find_dotenv, load_dotenv
from psycopg.rows import DictRow, dict_row
from storage import LocalFileStorage

from app import worker as worker_module
from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.nodes import score as score_module
from contracts import Candidate, LookCard

pytestmark = pytest.mark.enable_socket

REPO_ROOT = Path(__file__).resolve().parents[3]
TEST_SCHEMA = "comiccanvas_test_worker_jobs"

_FAKE_EMBEDDING = np.ones(384, dtype=np.float32) / np.sqrt(384)


def _database_url() -> str | None:
    load_dotenv(find_dotenv(usecwd=True))
    return os.environ.get("DATABASE_URL")


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    url = _database_url()
    if not url:
        pytest.skip("DATABASE_URL not configured; see the README for local Postgres setup")
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')
    env = {**os.environ, "DB_SCHEMA": TEST_SCHEMA}
    subprocess.run(["uv", "run", "alembic", "upgrade", "head"], cwd=REPO_ROOT, env=env, check=True)
    yield url
    with psycopg.connect(url, autocommit=True) as conn:
        conn.execute(f'DROP SCHEMA IF EXISTS "{TEST_SCHEMA}" CASCADE')


@pytest.fixture(autouse=True)
def _db_schema_env(monkeypatch: pytest.MonkeyPatch, database_url: str) -> None:
    monkeypatch.setenv("DB_SCHEMA", TEST_SCHEMA)


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_EXTRACTOR", "mock:mock-extractor")
    monkeypatch.setenv("LLM_SCRIPT", "mock:mock-script")
    monkeypatch.setenv("LLM_PROMPTS", "mock:mock-prompts")
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")

    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    panels = [
        {
            "id": i,
            "beat_id": "b1",
            "place": "kitchen",
            "time_of_day": "morning",
            "expression": "content",
            "action": action,
            "framing": framing,
            "caption_a": caption,
            "caption_b": "",
            "bubble": "",
            "cast": [],
        }
        for i, (action, framing, caption) in enumerate(
            [
                ("typing on a laptop", "medium", "Finally answered that email."),
                ("pouring coffee", "close", "Coffee number two."),
                ("walking outdoors", "wide", "Took the long way home."),
                ("lifting a dumbbell", "medium", "Leg day."),
            ],
            start=1,
        )
    ]
    provider.set_structured_response(
        "Script", json.dumps({"mood": "steady", "quiet_day": False, "panels": panels})
    )
    provider.set_structured_response(
        "PanelClauses",
        json.dumps(
            {
                "character_expression_clause": "content expression",
                "environment_clause": "a chipped mug, morning light, a wooden table, steam",
            }
        ),
    )


@pytest.fixture(autouse=True)
def _mock_scoring(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_identity(
        image_bytes: bytes, look_card: LookCard, **kwargs: object
    ) -> tuple[float, None]:
        return 0.9, None

    monkeypatch.setattr(score_module, "score_identity", fake_identity)
    monkeypatch.setattr(score_module, "embed_image", lambda *a, **k: _FAKE_EMBEDDING)
    monkeypatch.setattr(score_module, "score_style_from_embedding", lambda *a, **k: 0.7)

    async def fake_judge(panel: object, candidates: list[Candidate], **kwargs: object):  # type: ignore[no-untyped-def]
        return {c.id: {"judge_scene": 0.75, "judge_overall": 0.75} for c in candidates}, 0.001

    monkeypatch.setattr(score_module, "judge_panel", fake_judge)
    monkeypatch.setattr(score_module, "score_alignment", lambda *a, **k: 0.5)
    monkeypatch.setattr(score_module, "score_detail", lambda *a, **k: 0.3)
    monkeypatch.setattr(score_module, "has_detected_text", lambda *a, **k: False)


@pytest.fixture(autouse=True)
def _local_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        worker_module,
        "_storage",
        lambda: LocalFileStorage(root=tmp_path, base_url="http://localhost:8000"),
    )


def _conn(database_url: str) -> psycopg.Connection[DictRow]:
    conn = psycopg.connect(database_url, row_factory=dict_row)
    conn.execute(f'SET search_path TO "{TEST_SCHEMA}", public')
    return conn


def _seed_identity(database_url: str, *, user_id: str, person_id: str) -> None:
    look_card = {
        "hair": "short black hair",
        "glasses": "none",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "mustard sweater",
        "distinguishing": "none",
        "gender_term": "person",
    }
    # `master_path` is seeded as a full URL, matching what the real
    # `train_character_job` actually writes there (`Storage.put_object`
    # returns `get_url(key)`, a full URL, on both LocalFileStorage and
    # R2Storage) — `_load_identity` must use it directly, not run it back
    # through `storage.get_url()` a second time.
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id) VALUES (%s)", (user_id,))
        conn.execute(
            "INSERT INTO people (id, user_id, name) VALUES (%s, %s, 'Test Person')",
            (person_id, user_id),
        )
        conn.execute(
            "INSERT INTO identity_models (person_id, master_path, look_card) VALUES (%s, %s, %s)",
            (
                person_id,
                f"http://localhost:8000/media/people/{person_id}/master.png",
                json.dumps(look_card),
            ),
        )
        conn.commit()


def _seed_job(database_url: str, *, job_id: str, user_id: str) -> None:
    with _conn(database_url) as conn:
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'pending')",
            (job_id, user_id),
        )
        conn.commit()


def test_load_identity_does_not_double_prefix_a_stored_url(database_url: str) -> None:
    """`master_path` is stored as a full URL (see `_seed_identity`'s docstring); `_load_identity`
    must use it directly rather than running it back through `storage.get_url()`, which would
    prefix it a second time."""
    person_id = f"load-identity-test-{uuid4().hex[:8]}"
    user_id = f"load-identity-user-{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id=user_id, person_id=person_id)
    expected_url = f"http://localhost:8000/media/people/{person_id}/master.png"

    with _conn(database_url) as conn:
        identity, _look_card = worker_module._load_identity(conn, user_id)

    assert identity.master_url == expected_url


async def test_run_daily_job_persists_a_full_day_end_to_end(database_url: str) -> None:
    """`job_id` includes a fresh uuid: `open_day_graph`'s checkpointer lives in its own permanent
    `langgraph` schema (not test-isolated — see graph.py's own docstring on why it must be
    separate from the business-data schema), so a fixed job_id would resume a prior test run's
    checkpoint instead of executing fresh. Real per-user job_ids in production
    (`f"{user_id}:{date}"`, `app.routers.days`) don't have this problem since they're naturally
    unique per real day."""
    job_id = f"worker-test-user:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id="worker-test-user", person_id="worker-test-person")
    _seed_job(database_url, job_id=job_id, user_id="worker-test-user")

    await worker_module.run_daily_job(
        {},
        job_id,
        "worker-test-user",
        "2026-09-16",
        text="Answered emails, made coffee, walked home the long way, then hit the gym.",
    )

    with _conn(database_url) as conn:
        job = conn.execute(
            "SELECT status, error, events FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        assert job is not None
        assert job["status"] == "done", job["error"]
        assert [e["node"] for e in job["events"]] == [
            "input",
            "beats",
            "script",
            "prompts",
            "generate",
            "critic",
            "compose",
            "memory",
        ]

        day = conn.execute(
            "SELECT strip_url, story_url FROM days WHERE job_id = %s", (job_id,)
        ).fetchone()
        assert day is not None
        assert day["strip_url"] is not None
        assert day["story_url"] is not None

        panels = conn.execute(
            "SELECT panel_id FROM panels WHERE day_id = %s ORDER BY panel_id", (job_id,)
        ).fetchall()
        assert [p["panel_id"] for p in panels] == [1, 2, 3, 4]

        candidates = conn.execute(
            "SELECT panel_id, chosen, prompt_id FROM candidates WHERE day_id = %s", (job_id,)
        ).fetchall()
        chosen_panels = {c["panel_id"] for c in candidates if c["chosen"]}
        assert chosen_panels == {1, 2, 3, 4}
        assert all(c["prompt_id"] is not None for c in candidates)

        # Every candidate keeps its image embedding (for the personal ranking model) and
        # carries the vision judge's ratings.
        embedded = conn.execute(
            "SELECT embedding, scores FROM candidates WHERE day_id = %s", (job_id,)
        ).fetchall()
        assert len(embedded) == 12
        assert all(r["embedding"] is not None and len(r["embedding"]) == 384 for r in embedded)
        assert all(r["scores"]["judge_scene"] == 0.75 for r in embedded)

        # Every panel's three candidates differ on ONE axis (levels -1/0/+1), the measured
        # image features are stored next to the critic scores, and the chosen one records
        # whether it was an exploratory pick.
        scored = conn.execute(
            "SELECT panel_id, chosen, scores FROM candidates WHERE day_id = %s ORDER BY panel_id",
            (job_id,),
        ).fetchall()
        assert len(scored) == 12
        for panel in (1, 2, 3, 4):
            rows = [r for r in scored if r["panel_id"] == panel]
            assert sorted(r["scores"]["axis_level"] for r in rows) == [-1.0, 0.0, 1.0]
            assert len({r["scores"]["axis_id"] for r in rows}) == 1
        assert {r["scores"]["axis_id"] for r in scored} == {0.0, 1.0, 2.0}  # spread across axes
        # Every candidate also records ALL of its knob levels and its step from the default,
        # so the knob model can learn from it and a tap on the default can be told apart from
        # a tap on a variation.
        for r in scored:
            for key in ("level_warmth", "level_closeness", "level_expression", "phrased"):
                assert key in r["scores"]
        assert sorted(r["scores"]["offset"] for r in scored if r["panel_id"] == 1) == [
            -1.0,
            0.0,
            1.0,
        ]
        for r in scored:
            for feature in ("warmth", "brightness", "saturation", "contrast", "closeness"):
                assert feature in r["scores"]
            assert "reward" in r["scores"]
        assert all("explored" in r["scores"] for r in scored if r["chosen"])

        prompts = conn.execute(
            "SELECT panel_id, positive FROM image_prompts WHERE day_id = %s ORDER BY panel_id",
            (job_id,),
        ).fetchall()
        assert [p["panel_id"] for p in prompts] == [1, 2, 3, 4]
        assert all(p["positive"] for p in prompts)
        # `graph.py`'s `_prompts` node must pass `look_card` to `write_prompts`, so every
        # panel's positive prompt carries the person's real hair/face-shape traits
        # (`_seed_identity`'s look card) instead of `_describe_look_card(None)`'s generic
        # "a person" fallback.
        assert all("short black hair" in p["positive"] for p in prompts)


async def test_apply_pair_choice_job_flips_chosen_and_recomposes_the_strip(
    database_url: str, tmp_path: Path
) -> None:
    """`apply_pair_choice_job` must flip the tapped candidate's `chosen` flag (and only within
    its own panel) and actually recompose `strip.png` from it — checked here by comparing the
    real file bytes on disk before/after, not just trusting a status code."""
    job_id = f"worker-test-user-pick:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id="worker-test-user-pick", person_id="worker-test-pick")
    _seed_job(database_url, job_id=job_id, user_id="worker-test-user-pick")

    await worker_module.run_daily_job(
        {}, job_id, "worker-test-user-pick", "2026-09-16", text="A perfectly ordinary day."
    )

    with _conn(database_url) as conn:
        panel_1_before = conn.execute(
            "SELECT id, url, chosen FROM candidates WHERE day_id = %s AND panel_id = 1", (job_id,)
        ).fetchall()
        runner_up = next(c for c in panel_1_before if not c["chosen"])
        critic_pick = next(c for c in panel_1_before if c["chosen"])
        # What `POST /feedback/pair` writes for an A/B tap (the API isn't run here).
        conn.execute(
            "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source) "
            "VALUES (%s, 1, %s, %s, 'ab')",
            (job_id, runner_up["url"], critic_pick["url"]),
        )
        conn.commit()

        panel_2_chosen_before = conn.execute(
            "SELECT id FROM candidates WHERE day_id = %s AND panel_id = 2 AND chosen", (job_id,)
        ).fetchone()

    strip_path = tmp_path / "strips" / "worker-test-user-pick" / "2026-09-16" / "strip.png"
    bytes_before = strip_path.read_bytes()

    await worker_module.apply_pair_choice_job(
        {}, job_id, "worker-test-user-pick", 1, runner_up["url"]
    )

    with _conn(database_url) as conn:
        panel_1_after = conn.execute(
            "SELECT id, chosen FROM candidates WHERE day_id = %s AND panel_id = 1", (job_id,)
        ).fetchall()
        chosen_after = next(c for c in panel_1_after if c["chosen"])
        assert chosen_after["id"] == runner_up["id"]  # the tapped candidate is now chosen
        # ...and the tap taught the preference model (snapshot + model row).
        snap = conn.execute(
            "SELECT tap_index, prob FROM preference_snapshots WHERE user_id = %s",
            ("worker-test-user-pick",),
        ).fetchall()
        assert [s["tap_index"] for s in snap] == [1]
        model_row = conn.execute(
            "SELECT n_taps, active FROM preference_models WHERE user_id = %s",
            ("worker-test-user-pick",),
        ).fetchone()
        assert model_row is not None and model_row["n_taps"] == 1 and not model_row["active"]
        assert sum(c["chosen"] for c in panel_1_after) == 1  # exactly one chosen per panel

        panel_2_chosen_after = conn.execute(
            "SELECT id FROM candidates WHERE day_id = %s AND panel_id = 2 AND chosen", (job_id,)
        ).fetchone()
        assert panel_2_chosen_after is not None
        assert panel_2_chosen_before is not None
        assert panel_2_chosen_after["id"] == panel_2_chosen_before["id"]  # untouched

    bytes_after = strip_path.read_bytes()
    assert bytes_after != bytes_before  # the strip was genuinely recomposed


async def test_apply_pair_choice_job_does_not_revert_an_unrelated_regenerated_panel(
    database_url: str,
) -> None:
    """`apply_pair_choice_job` must only touch the panel it's given, reading current candidates
    from the relational tables rather than the job's (possibly stale) LangGraph checkpoint —
    `regenerate_panel_job` only ever writes fresh candidates to the relational tables, never back
    into the checkpoint, so persisting the checkpoint's view of an already-regenerated other panel
    would resurrect its old `chosen` candidate. Regenerating panel 2, then applying an A/B pick on
    panel 1, must leave panel 2 with exactly the regenerate's own result, untouched."""
    job_id = f"worker-test-user-cross:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id="worker-test-user-cross", person_id="worker-test-cross")
    _seed_job(database_url, job_id=job_id, user_id="worker-test-user-cross")

    await worker_module.run_daily_job(
        {}, job_id, "worker-test-user-cross", "2026-09-16", text="An ordinary day."
    )

    await worker_module.regenerate_panel_job(
        {}, job_id, "worker-test-user-cross", 2, "make it different"
    )

    with _conn(database_url) as conn:
        panel_2_after_regenerate = conn.execute(
            "SELECT id FROM candidates WHERE day_id = %s AND panel_id = 2 AND chosen", (job_id,)
        ).fetchone()
        assert panel_2_after_regenerate is not None

        panel_1_candidates = conn.execute(
            "SELECT id, url, chosen FROM candidates WHERE day_id = %s AND panel_id = 1", (job_id,)
        ).fetchall()
        runner_up = next(c for c in panel_1_candidates if not c["chosen"])

    await worker_module.apply_pair_choice_job(
        {}, job_id, "worker-test-user-cross", 1, runner_up["url"]
    )

    with _conn(database_url) as conn:
        panel_2_after_pick = conn.execute(
            "SELECT id, chosen FROM candidates WHERE day_id = %s AND panel_id = 2", (job_id,)
        ).fetchall()
        chosen_ids = [c["id"] for c in panel_2_after_pick if c["chosen"]]
        assert chosen_ids == [panel_2_after_regenerate["id"]]  # unchanged, and still only one


async def test_run_daily_job_redo_actually_reruns_instead_of_resuming_to_a_noop(
    database_url: str,
) -> None:
    """"Redo today's strip" re-calls `POST /days` for the same date, which reuses the same
    deterministic `job_id` — LangGraph would otherwise see a checkpoint thread already at its END
    state and resume straight there with nothing left to run. `run_daily_job`'s `is_redo` check (a
    `days` row already existing for this `job_id`) must call `day_graph.reset()` so the second run
    actually re-executes every node against the new text."""
    job_id = f"worker-test-user-redo:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id="worker-test-user-redo", person_id="worker-test-redo")
    _seed_job(database_url, job_id=job_id, user_id="worker-test-user-redo")

    await worker_module.run_daily_job(
        {}, job_id, "worker-test-user-redo", "2026-09-16", text="First pass, version one."
    )

    # Simulate what `POST /days` does on a redo: reset the job row back to
    # pending, then re-invoke the worker with fresh text for the same job_id.
    with _conn(database_url) as conn:
        conn.execute(
            "UPDATE jobs SET status = 'pending', error = NULL, events = '[]' WHERE id = %s",
            (job_id,),
        )
        conn.commit()

    await worker_module.run_daily_job(
        {}, job_id, "worker-test-user-redo", "2026-09-16", text="Second pass, totally different."
    )

    with _conn(database_url) as conn:
        job = conn.execute(
            "SELECT status, error, events FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        assert job is not None
        assert job["status"] == "done", job["error"]
        # A no-op resume would yield zero node events on the second run;
        # a genuine redo re-executes the whole pipeline again.
        assert [e["node"] for e in job["events"]] == [
            "input",
            "beats",
            "script",
            "prompts",
            "generate",
            "critic",
            "compose",
            "memory",
        ]

        day = conn.execute(
            "SELECT text, strip_url FROM days WHERE job_id = %s", (job_id,)
        ).fetchone()
        assert day is not None
        assert day["text"] == "Second pass, totally different."
        assert day["strip_url"] is not None


async def test_run_daily_job_fails_cleanly_with_no_approved_character(database_url: str) -> None:
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id) VALUES ('worker-test-user-2')")
        conn.commit()
    _seed_job(database_url, job_id="worker-test-user-2:2026-09-16", user_id="worker-test-user-2")

    await worker_module.run_daily_job(
        {}, "worker-test-user-2:2026-09-16", "worker-test-user-2", "2026-09-16", text="hi"
    )

    with _conn(database_url) as conn:
        job = conn.execute(
            "SELECT status, error FROM jobs WHERE id = %s",
            ("worker-test-user-2:2026-09-16",),
        ).fetchone()
        assert job is not None
        assert job["status"] == "failed"
        assert "no approved character" in job["error"]


def _seed_week_of_panels(
    database_url: str, user_id: str, iso_week: str, *, days: int, panels_per_day: int = 4
) -> list[str]:
    """`days` daily strips (from Monday), each with panels that have a kept (chosen) image, plus
    an ordinary diary entry on the week's Sunday. Returns the kept images' urls."""
    year, week = (int(x) for x in iso_week.split("-W"))
    urls = []
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (user_id,))
        for offset in range(days):
            day = date.fromisocalendar(year, week, offset + 1)
            job = f"{user_id}:{day.isoformat()}"
            conn.execute(
                "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'daily', 'done')",
                (job, user_id),
            )
            conn.execute(
                "INSERT INTO days (job_id, user_id, date, source, mood) "
                "VALUES (%s,%s,%s,'text',%s)",
                (job, user_id, day, "content"),
            )
            for panel in range(1, panels_per_day + 1):
                conn.execute(
                    "INSERT INTO panels (day_id, panel_id, place, time_of_day, expression, "
                    "action, framing, caption_a) VALUES (%s,%s,'desk','morning','content',%s,"
                    "'medium',%s)",
                    (job, panel, f"scene {offset}-{panel}", f"Caption {offset}-{panel}"),
                )
                url = f"http://x/media/{job}/kept-{panel}.png"
                urls.append(url)
                conn.execute(
                    "INSERT INTO candidates (id, day_id, panel_id, url, seed, chosen) "
                    "VALUES (%s,%s,%s,%s,1,true), (%s,%s,%s,%s,2,false)",
                    (f"{job}-{panel}a", job, panel, url, f"{job}-{panel}b", job, panel, url + "x"),
                )
        conn.commit()
    return urls


def _canned_highlights(refs: list[int]) -> str:
    return json.dumps(
        {
            "mood": "busy but good",
            "highlights": [{"ref": r, "caption": f"Highlight {r}"} for r in refs],
        }
    )


async def test_the_weekly_recap_picks_highlights_from_existing_panels_and_draws_nothing(
    database_url: str,
) -> None:
    """No images for the recap — a language model chooses the week's highlights and the recap is
    those existing pictures in the six-panel grid. The recap row coexists with a Sunday diary
    entry, regenerating replaces it, and nothing is generated."""
    user_id = "worker-test-weekly-highlights"
    iso_week = "2026-W38"
    _seed_week_of_panels(database_url, user_id, iso_week, days=7)
    weekly_job_id = f"{user_id}:weekly:{iso_week}"
    _seed_job(database_url, job_id=weekly_job_id, user_id=user_id)
    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("WeekHighlights", _canned_highlights([1, 6, 10, 15, 20, 26]))

    for _ in range(2):  # the second run is a regenerate
        await worker_module.run_weekly_job({}, weekly_job_id, user_id, iso_week)
        with _conn(database_url) as conn:
            job = conn.execute(
                "SELECT status, error FROM jobs WHERE id = %s", (weekly_job_id,)
            ).fetchone()
        assert job is not None and job["status"] == "done", job

    with _conn(database_url) as conn:
        weekly = conn.execute(
            "SELECT mood, kind FROM days WHERE job_id = %s", (weekly_job_id,)
        ).fetchone()
        panels = conn.execute(
            "SELECT panel_id, caption_a FROM panels WHERE day_id = %s ORDER BY panel_id",
            (weekly_job_id,),
        ).fetchall()
        kept = conn.execute(
            "SELECT c.panel_id, c.url FROM candidates c WHERE c.day_id = %s AND c.chosen "
            "ORDER BY c.panel_id",
            (weekly_job_id,),
        ).fetchall()
        sunday_rows = conn.execute(
            "SELECT kind FROM days WHERE user_id = %s AND date = %s ORDER BY kind",
            (user_id, date.fromisocalendar(2026, 38, 7)),
        ).fetchall()
    assert weekly is not None and (weekly["mood"], weekly["kind"]) == ("busy but good", "weekly")
    assert [p["caption_a"] for p in panels] == [f"Highlight {r}" for r in (1, 6, 10, 15, 20, 26)]
    # every recap panel shows an image the person already kept for a daily panel
    assert len(kept) == 6 and all(k["url"].endswith(".png") and "kept-" in k["url"] for k in kept)
    assert [r["kind"] for r in sunday_rows] == ["daily", "weekly"]  # a Sunday entry and the recap


async def test_a_week_with_few_panels_still_makes_a_recap_and_an_empty_week_fails_cleanly(
    database_url: str,
) -> None:
    user_id = "worker-test-weekly-few"
    iso_week = "2026-W38"
    _seed_week_of_panels(database_url, user_id, iso_week, days=1, panels_per_day=2)
    job = f"{user_id}:weekly:{iso_week}"
    _seed_job(database_url, job_id=job, user_id=user_id)
    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("WeekHighlights", _canned_highlights([0, 1]))
    await worker_module.run_weekly_job({}, job, user_id, iso_week)
    with _conn(database_url) as conn:
        n = conn.execute("SELECT count(*) AS n FROM panels WHERE day_id = %s", (job,)).fetchone()
    assert n is not None and n["n"] == 2  # as many as exist

    empty_user = "worker-test-weekly-empty"
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id) VALUES (%s) ON CONFLICT DO NOTHING", (empty_user,))
        conn.commit()
    empty_job = f"{empty_user}:weekly:{iso_week}"
    _seed_job(database_url, job_id=empty_job, user_id=empty_user)
    await worker_module.run_weekly_job({}, empty_job, empty_user, iso_week)
    with _conn(database_url) as conn:
        failed = conn.execute(
            "SELECT status, error FROM jobs WHERE id = %s", (empty_job,)
        ).fetchone()
    assert (
        failed is not None and failed["status"] == "failed" and "no comic panels" in failed["error"]
    )


def test_worker_settings_job_timeout_is_generous() -> None:
    """arq's own default `job_timeout` (300s) is shorter than a realistic worst case for
    `run_daily_job` (up to 12 base candidates plus up to 2 identity-gate retries per panel, each
    a real `fal-ai/flux-2/edit` call with its own queue+poll latency). When arq's watchdog fires,
    it cancels the task via `asyncio.CancelledError`, which is not an `Exception` subclass, so
    `run_daily_job`'s own `except Exception` never runs and the job would be left stuck at
    `status='running'` forever without a much larger timeout."""
    assert worker_module.WorkerSettings.job_timeout >= 1800


async def test_train_character_job_always_ends_on_a_terminal_status(database_url: str) -> None:
    """The job owns a real `train:{person_id}` `jobs` row — a failure (here: a person with no
    processed photos at all) must land on that row as `failed` with a real message, never leave
    it stuck at `pending`/`running`."""
    user_id = f"train-user-{uuid4().hex[:8]}"
    person_id = f"train-person-{uuid4().hex[:8]}"
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id, email) VALUES (%s, NULL)", (user_id,))
        conn.execute(
            "INSERT INTO people (id, user_id, name) VALUES (%s, %s, 'T')", (person_id, user_id)
        )
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'train', 'pending')",
            (worker_module.train_job_id(person_id), user_id),
        )
        conn.commit()

    await worker_module.train_character_job({}, person_id)

    with _conn(database_url) as conn:
        job = conn.execute(
            "SELECT status, error FROM jobs WHERE id = %s", (worker_module.train_job_id(person_id),)
        ).fetchone()
    assert job is not None
    assert job["status"] == "failed"
    assert job["error"]


async def test_train_character_job_saves_top_candidates_for_the_person_to_pick(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Training doesn't auto-approve candidate #1 (and discard the rest) — it saves the top
    candidates under `people/{id}/master_candidates/` and records them on
    `identity_models.master_candidates` for the person to choose from; `master_path` must stay
    untouched until they pick. The paid fal.ai call and face detection are faked — this covers
    the worker's own wiring, at zero spend."""
    from types import SimpleNamespace

    from contracts import FeatureChecklistResult, MasterCandidate

    user_id = f"pick-user-{uuid4().hex[:8]}"
    person_id = f"pick-person-{uuid4().hex[:8]}"
    look_card = {
        "hair": "short black hair",
        "glasses": "none",
        "skin_tone": "medium",
        "face_shape": "oval",
        "signature_outfit": "sweater",
        "distinguishing": "none",
        "gender_term": "man",
    }
    (tmp_path / "people" / person_id / "raw").mkdir(parents=True)
    (tmp_path / "people" / person_id / "raw" / "p.jpg").write_bytes(b"x")
    with _conn(database_url) as conn:
        conn.execute("INSERT INTO users (id, email) VALUES (%s, NULL)", (user_id,))
        conn.execute(
            "INSERT INTO people (id, user_id, name, gender_term, age) "
            "VALUES (%s, %s, 'P', 'man', 34)",
            (person_id, user_id),
        )
        conn.execute(
            "INSERT INTO identity_models (person_id, source_photo_count, look_card) "
            "VALUES (%s, 1, %s::jsonb)",
            (person_id, json.dumps(look_card)),
        )
        conn.execute(
            "INSERT INTO person_photos (person_id, storage_key, status) VALUES (%s, %s, 'usable')",
            (person_id, f"people/{person_id}/raw/p.jpg"),
        )
        conn.execute(
            "INSERT INTO jobs (id, user_id, kind, status) VALUES (%s, %s, 'train', 'pending')",
            (worker_module.train_job_id(person_id), user_id),
        )
        conn.commit()

    checklist = FeatureChecklistResult(
        checks=[],
        score=1.0,
        mandatory_passed=True,
        artifacts_clean=True,
        image_sha256="",
        model="mock",
    )
    ranked = [
        MasterCandidate(
            id=f"{i}-{i}",
            url="",
            seed=100 + i,
            source_crop_index=0,
            prompt="p",
            checklist=checklist,
            style_score=0.5,
            arcface_agreement=None,
            rank_score=1.0 - i / 10,
        )
        for i in range(3)
    ]

    async def fake_design_master(request: object, *, confirm: bool) -> object:
        return ranked, ranked, {c.id: f"img-{c.id}".encode() for c in ranked}

    monkeypatch.setattr(
        worker_module, "process_batch", lambda paths: [SimpleNamespace(status="usable")]
    )
    monkeypatch.setattr(worker_module, "select_best_crops", lambda usable: [b"crop"])
    monkeypatch.setattr(worker_module, "design_master", fake_design_master)

    await worker_module.train_character_job({}, person_id)

    with _conn(database_url) as conn:
        identity = conn.execute(
            "SELECT master_path, master_candidates, look_card FROM identity_models "
            "WHERE person_id = %s",
            (person_id,),
        ).fetchone()
        job = conn.execute(
            "SELECT status, events FROM jobs WHERE id = %s",
            (worker_module.train_job_id(person_id),),
        ).fetchone()
    assert identity is not None and job is not None
    assert identity["master_path"] is None  # nothing auto-approved
    # Age (and gender) answered in the wizard land on the look card even when it already
    # existed, so every panel says "a 34-year-old man".
    assert identity["look_card"]["age_descriptor"] == "34-year-old"
    assert identity["look_card"]["gender_term"] == "man"
    assert [c["id"] for c in identity["master_candidates"]] == ["0-0", "1-1", "2-2"]
    for c in identity["master_candidates"]:
        assert (tmp_path / "people" / person_id / "master_candidates" / f"{c['id']}.png").exists()
    assert job["status"] == "done"
    assert job["events"][-1]["node"] == "review"


async def test_regenerate_uses_the_note_and_fresh_seeds_and_recomposes_the_strip(
    database_url: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A regenerate note must actually reach the prompt (not just be logged), each press must get
    a fresh seed the panel hasn't used (so the image model can't just return the same picture),
    and the composed strip must be rebuilt afterward."""
    user = "worker-test-user-regen"
    job_id = f"{user}:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id=user, person_id="worker-test-regen")
    _seed_job(database_url, job_id=job_id, user_id=user)
    await worker_module.run_daily_job({}, job_id, user, "2026-09-16", text="An ordinary day.")

    seen: list[tuple[str, int]] = []
    from app.nodes.generate import generate_one_panel as real_generate

    async def spy(prompt: object, **kwargs: object) -> object:
        seen.append((prompt.positive, prompt.seed))  # type: ignore[attr-defined]
        return await real_generate(prompt, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(worker_module, "generate_one_panel", spy)

    with _conn(database_url) as conn:
        original_seeds = {
            r["seed"]
            for r in conn.execute(
                "SELECT seed FROM candidates WHERE day_id = %s AND panel_id = 2", (job_id,)
            ).fetchall()
        }
        first_pick = conn.execute(
            "SELECT id FROM candidates WHERE day_id = %s AND panel_id = 2 AND chosen", (job_id,)
        ).fetchone()
    strip = tmp_path / "strips" / user / "2026-09-16" / "strip.png"
    strip_before = strip.read_bytes()

    await worker_module.regenerate_panel_job({}, job_id, user, 2, "make me look less tired")
    await worker_module.regenerate_panel_job({}, job_id, user, 2, "")

    (with_note, seed_one), (without_note, seed_two) = seen[0], seen[-1]
    assert "Important change requested by the person: make me look less tired." in with_note
    assert "Important change" not in without_note  # an empty note adds nothing
    assert seed_one not in original_seeds  # a seed this panel hasn't used
    assert seed_two != seed_one  # and every press gets a different one

    with _conn(database_url) as conn:
        chosen = conn.execute(
            "SELECT id FROM candidates WHERE day_id = %s AND panel_id = 2 AND chosen", (job_id,)
        ).fetchall()
    assert len(chosen) == 1 and first_pick is not None and chosen[0]["id"] != first_pick["id"]
    assert strip.read_bytes() != strip_before  # the composed strip shows the new panel


async def test_an_edited_caption_survives_a_later_tap_and_reaches_the_strip(
    database_url: str, tmp_path: Path
) -> None:
    """A saved caption edit must survive a later A/B tap (which re-saves the day from its
    LangGraph checkpoint — that checkpoint still holds the original captions, so it must not
    silently put them back), and `recompose_strip_job` must bake the edit into the strip."""
    user = "worker-test-user-caption"
    job_id = f"{user}:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id=user, person_id="worker-test-caption")
    _seed_job(database_url, job_id=job_id, user_id=user)
    await worker_module.run_daily_job({}, job_id, user, "2026-09-16", text="An ordinary day.")
    strip = tmp_path / "strips" / user / "2026-09-16" / "strip.png"
    before_edit = strip.read_bytes()

    with _conn(database_url) as conn:  # what `POST /feedback/caption` writes
        conn.execute(
            "UPDATE panels SET caption_a = 'MY OWN WORDS' WHERE day_id = %s AND panel_id = 1",
            (job_id,),
        )
        conn.commit()

    await worker_module.recompose_strip_job({}, job_id, user)
    after_recompose = strip.read_bytes()
    assert after_recompose != before_edit  # the strip file carries the new caption

    with _conn(database_url) as conn:
        runner_up = conn.execute(
            "SELECT url FROM candidates WHERE day_id = %s AND panel_id = 2 AND NOT chosen LIMIT 1",
            (job_id,),
        ).fetchone()
    assert runner_up is not None
    await worker_module.apply_pair_choice_job({}, job_id, user, 2, runner_up["url"])

    with _conn(database_url) as conn:
        caption = conn.execute(
            "SELECT caption_a FROM panels WHERE day_id = %s AND panel_id = 1", (job_id,)
        ).fetchone()
    assert caption is not None and caption["caption_a"] == "MY OWN WORDS"


async def test_a_redo_replaces_the_previous_runs_images_picks_and_ratings(
    database_url: str,
) -> None:
    """A redo must not leave the previous run's candidates in the database beside the new ones —
    otherwise the picker mixes two batches and the learning model keeps counting picks (and
    ratings) on images that no longer exist."""
    user = "worker-test-user-redo2"
    job_id = f"{user}:2026-09-16:{uuid4().hex[:8]}"
    _seed_identity(database_url, user_id=user, person_id="worker-test-redo2")
    _seed_job(database_url, job_id=job_id, user_id=user)
    await worker_module.run_daily_job({}, job_id, user, "2026-09-16", text="First pass.")

    with _conn(database_url) as conn:
        conn.execute(
            "INSERT INTO candidates (id, day_id, panel_id, url, seed, scores) "
            "VALUES ('old-run', %s, 1, 'http://x/old.png', 5, '{}')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO image_pairs (day_id, panel_id, chosen, rejected, source, tap_id) "
            "VALUES (%s, 1, 'http://x/old.png', 'http://x/older.png', 'ab', 't1')",
            (job_id,),
        )
        conn.execute(
            "INSERT INTO image_ratings (day_id, panel_id, url, rating) "
            "VALUES (%s, 1, 'http://x/old.png', 2)",
            (job_id,),
        )
        conn.execute("UPDATE jobs SET status = 'pending', events = '[]' WHERE id = %s", (job_id,))
        conn.commit()

    await worker_module.run_daily_job({}, job_id, user, "2026-09-16", text="Second pass.")

    with _conn(database_url) as conn:
        n_candidates = conn.execute(
            "SELECT count(*) AS n FROM candidates WHERE day_id = %s", (job_id,)
        ).fetchone()
        stale = conn.execute(
            "SELECT (SELECT count(*) FROM candidates WHERE id = 'old-run') AS c, "
            "(SELECT count(*) FROM image_pairs WHERE day_id = %s) AS p, "
            "(SELECT count(*) FROM image_ratings WHERE day_id = %s) AS r",
            (job_id, job_id),
        ).fetchone()
    assert n_candidates is not None and n_candidates["n"] == 12  # this run's, not 13 or 24
    assert stale is not None and (stale["c"], stale["p"], stale["r"]) == (0, 0, 0)
