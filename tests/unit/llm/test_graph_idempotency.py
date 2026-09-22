"""Idempotency — re-running a `job_id` resumes from the last completed
node using persisted `DayState`, via a real Postgres-backed LangGraph
checkpointer (`open_day_graph`).

Runs against the real local Postgres (`DATABASE_URL`) in a dedicated
`comiccanvas_test` schema, mirroring `test_memory_postgres.py`'s
real-infra pattern — skips if `DATABASE_URL` isn't configured.
`enable_socket` is for the real Postgres TCP connection; the model calls
used here are the same mocks `test_graph.py` already established.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError
from storage import LocalFileStorage

from app import graph as graph_module
from app.config import get_settings
from app.db import drop_schema
from app.images.base import IdentityRef
from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore
from app.nodes import score as score_module
from app.nodes.beats import extract_beats as real_extract_beats
from app.nodes.script import write_script as real_write_script
from contracts import Candidate, DayState, LookCard

pytestmark = pytest.mark.enable_socket

TEST_SCHEMA = "comiccanvas_test"

_FAKE_EMBEDDING = np.ones(384, dtype=np.float32) / np.sqrt(384)


def _require_database() -> None:
    try:
        get_settings()
    except ValidationError:
        pytest.skip("DATABASE_URL not configured; see the README for local Postgres setup")


def _look_card() -> LookCard:
    return LookCard(
        hair="short black wavy hair",
        glasses="black rectangular glasses",
        skin_tone="medium",
        face_shape="oval",
        signature_outfit="mustard sweater",
        distinguishing="a small mole above the left eyebrow",
        gender_term="man",
    )


def _script_json() -> str:
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
    return json.dumps({"mood": "steady", "quiet_day": False, "panels": panels})


def _clauses_json() -> str:
    return json.dumps(
        {
            "character_expression_clause": "content expression",
            "environment_clause": "a chipped mug, morning light, a wooden table, steam",
        }
    )


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_EXTRACTOR", "mock:mock-extractor")
    monkeypatch.setenv("LLM_SCRIPT", "mock:mock-script")
    monkeypatch.setenv("LLM_PROMPTS", "mock:mock-prompts")
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")

    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("Script", _script_json())
    provider.set_structured_response("PanelClauses", _clauses_json())


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
def _clean_schema() -> Iterator[None]:
    _require_database()
    drop_schema(TEST_SCHEMA)
    yield
    drop_schema(TEST_SCHEMA)


def _day_state(job_id: str, *, day: date = date(2026, 3, 1)) -> DayState:
    return DayState(
        job_id=job_id,
        user_id="u1",
        date=day,
        source="text",
        text="Answered emails, made coffee, walked home the long way, then hit the gym.",
    )


async def test_a_killed_job_resumes_without_a_second_extractor_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kill after `script`, rerun: no second extractor call."""
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()

    extractor_calls = 0

    async def counting_extract_beats(state: DayState, **kwargs: object) -> DayState:
        nonlocal extractor_calls
        extractor_calls += 1
        return await real_extract_beats(state, **kwargs)  # type: ignore[arg-type]

    script_calls = 0

    async def failing_write_script(state: DayState, **kwargs: object) -> DayState:
        nonlocal script_calls
        script_calls += 1
        if script_calls == 1:
            raise RuntimeError("simulated kill during script")
        return await real_write_script(state, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(graph_module, "extract_beats", counting_extract_beats)
    monkeypatch.setattr(graph_module, "write_script", failing_write_script)

    async with graph_module.open_day_graph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=memory,
        schema=TEST_SCHEMA,
    ) as day_graph:
        job_id = "job-idempotency-1"

        with pytest.raises(RuntimeError, match="simulated kill during script"):
            await day_graph.run(_day_state(job_id))

        assert extractor_calls == 1
        assert script_calls == 1

        result = await day_graph.run(_day_state(job_id))

        assert extractor_calls == 1  # NOT re-run
        assert script_calls == 2  # re-run (it's the one that failed)
        assert result.strip_url is not None
        assert result.script is not None
        assert len(result.script.panels) == 4


async def test_a_brand_new_job_id_runs_fresh_with_a_checkpointer_configured(
    tmp_path: Path,
) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    async with graph_module.open_day_graph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
        schema=TEST_SCHEMA,
    ) as day_graph:
        result = await day_graph.run(_day_state("job-idempotency-fresh"))

    assert result.strip_url is not None
    assert result.script is not None
    assert len(result.script.panels) == 4


async def test_two_different_job_ids_do_not_interfere(tmp_path: Path) -> None:
    """Different checkpoint threads (`job_id`s) must not share resume
    state — each gets its own fresh run. Uses two different dates so
    the two runs also produce distinct storage keys (`compose_day`'s
    key is user+date scoped, matching "one strip per day," not
    job_id-scoped) — a same-user-and-date pair would collide on
    `strip_url` for a completely unrelated, correct reason."""
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    async with graph_module.open_day_graph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
        schema=TEST_SCHEMA,
    ) as day_graph:
        result_a = await day_graph.run(_day_state("job-idempotency-a", day=date(2026, 3, 1)))
        result_b = await day_graph.run(_day_state("job-idempotency-b", day=date(2026, 3, 2)))

    assert result_a.job_id == "job-idempotency-a"
    assert result_b.job_id == "job-idempotency-b"
    assert result_a.strip_url != result_b.strip_url
