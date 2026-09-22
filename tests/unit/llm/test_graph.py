"""graph.py — the daily pipeline (input -> beats -> script -> prompts ->
generate -> critic -> compose -> memory) as one connected `DayGraph`,
with every provider mocked (LLM, image, and the four critic scoring
functions — those are unit-tested against real models elsewhere; this
test is about orchestration, not re-verifying scoring accuracy).
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from storage import LocalFileStorage

from app import graph as graph_module
from app.images.base import IdentityRef
from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore
from app.nodes import score as score_module
from contracts import Candidate, DayState, LookCard

_FAKE_EMBEDDING = np.ones(384, dtype=np.float32) / np.sqrt(384)


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
    """The four critic signals + OCR guard are unit-tested against real
    models elsewhere — mocked here at `nodes/score.py`'s call boundary
    (patched where they're imported to, not where they're defined) so
    this test exercises real orchestration without paying for real model
    loads on every graph run. Scores clear `IDENTITY_THRESHOLD` so the
    happy path completes without retries."""

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


def _day_state() -> DayState:
    return DayState(
        job_id="job-graph-1",
        user_id="u1",
        date=date(2026, 3, 1),
        source="text",
        text="Answered emails, made coffee, walked home the long way, then hit the gym.",
    )


async def test_day_graph_runs_end_to_end_and_produces_a_4_panel_layout(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )

    result = await day_graph.run(_day_state())

    assert result.strip_url is not None
    assert result.layout is not None
    assert len(result.layout["strip"]["panels"]) == 4
    assert result.script is not None
    assert len(result.script.panels) == 4

    chosen = [c for c in result.candidates if c.chosen]
    assert {c.panel_id for c in chosen} == {1, 2, 3, 4}  # exactly one winner per panel
    assert all(c.scores.get("identity", 0.0) >= 0.55 for c in chosen)  # real gate, not skipped


async def test_day_graph_records_feature_flags_into_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WRITER/PROMPTER/REWARD_HEAD default to base/base/hand when unset;
    GENERATOR mirrors the real IMAGE_PROVIDER switch rather than being a
    second, independent flag (see graph.py's own docstring)."""
    monkeypatch.setenv("WRITER", "dpo")
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )

    result = await day_graph.run(_day_state())

    assert result.versions["writer_flag"] == "dpo"
    assert result.versions["prompter_flag"] == "base"
    assert result.versions["reward_head_flag"] == "hand"
    assert result.versions["generator_flag"] == "mock"


async def test_day_graph_completes_in_under_3_seconds(tmp_path: Path) -> None:
    import time

    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )

    start = time.monotonic()
    await day_graph.run(_day_state())
    elapsed = time.monotonic() - start

    assert elapsed < 3.0


async def test_day_graph_records_per_node_timing_in_versions(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )

    result = await day_graph.run(_day_state())

    for node in graph_module.NODE_ORDER:
        assert f"{node}_ms" in result.versions


async def test_day_graph_rejects_audio_source_not_yet_supported(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )
    state = DayState(
        job_id="job-audio-1",
        user_id="u1",
        date=date(2026, 3, 1),
        source="audio",
        audio_url="https://example.test/memo.webm",
    )

    with pytest.raises(graph_module.AsrNotImplementedError):
        await day_graph.run(state)


async def test_day_graph_writes_beats_to_memory_with_panel_urls(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=memory,
    )

    await day_graph.run(_day_state())

    # The mock LLM extractor always returns its own default canned
    # `BeatSheet` (see mock_provider.py's `_DEFAULT_BEAT_SHEET`) rather
    # than anything derived from `_day_state()`'s real text — "sunset"
    # is what that canned beat's event actually contains.
    results = memory.search("u1", "sunset")
    assert any(r.panel_urls for r in results)


async def test_day_graph_records_the_retry_prompt_not_the_discarded_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A panel that fails the identity gate once and passes on retry must end up with its last
    (surviving) prompt in `result.prompts`, not the original — a retry discards the failed batch
    entirely, and `persist.py` relies on this to link the right prompt to the right candidate."""
    call_count = {"panel_1": 0}

    async def flaky_identity(
        image_bytes: bytes, look_card: LookCard, **kwargs: object
    ) -> tuple[float, None]:
        call_count["panel_1"] += 1
        # First call for panel 1's original batch fails the gate; every
        # later call (the retry, and every other panel) passes.
        if call_count["panel_1"] <= 3:  # 3 candidates in the original batch
            return 0.1, None
        return 0.9, None

    monkeypatch.setattr(score_module, "score_identity", flaky_identity)

    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    day_graph = graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )

    result = await day_graph.run(_day_state())

    # `state.prompts` keeps BOTH the original (discarded) and the retry's
    # prompt for panel 1 — `persist.py` is what picks the LAST one per
    # panel_id as the one that actually produced the surviving
    # candidates; this test just confirms the retry's prompt genuinely
    # made it into the list at all (previously, it was lost entirely).
    panel_1_prompts = [p for p in result.prompts if p.panel_id == 1]
    assert len(panel_1_prompts) == 2  # original (discarded) + retry (kept)

    chosen = next(c for c in result.candidates if c.panel_id == 1 and c.chosen)
    assert chosen.scores["identity"] >= 0.55  # only the retry's batch could have passed
    assert result.errors == []  # a late pass on retry is not an error
