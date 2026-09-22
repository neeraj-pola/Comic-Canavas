"""Quiet-day and sensitive-day paths through the real `DayGraph`.

With every LLM role mocked, a "script correctly suppresses humor for a
sensitive day" claim can't actually be tested here — that's real model
behavior, covered by `ml/evals/script.py`'s judge eval and real
sensitive-flagging accuracy. What this does verify: a `BeatSheet`
genuinely flagged `too_short`/`sensitive` flows through `extract_beats`
unchanged, and a 2-panel, humor-appropriate `Script` flows all the way
through generate, critic, and compose without the pipeline assuming 4
panels anywhere — `compose.py`'s `_grid_shape` and the critic's per-panel
loop both need to genuinely handle panel count 2, not just 4.
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


def _clauses_json() -> str:
    return json.dumps(
        {
            "character_expression_clause": "quiet, gentle expression",
            "environment_clause": "a soft lamp, a folded blanket, a quiet room, low light",
        }
    )


def _two_panel_script_json(*, mood: str, captions: list[str]) -> str:
    panels = [
        {
            "id": i,
            "beat_id": "b1",
            "place": "bedroom",
            "time_of_day": "evening",
            "expression": "content",
            "action": action,
            "framing": framing,
            "caption_a": caption,
            "caption_b": "",
            "bubble": "",  # no punchline bubble — matches a suppressed-humor day
            "cast": [],
        }
        for i, (action, framing, caption) in enumerate(
            zip(
                ["sitting quietly", "looking out the window"],
                ["wide", "close"],
                captions,
                strict=True,
            ),
            start=1,
        )
    ]
    return json.dumps({"mood": mood, "quiet_day": True, "panels": panels})


def _beat_sheet_json(*, flags: list[str]) -> str:
    return json.dumps(
        {
            "date": "2026-03-01",
            "mood_arc": ["quiet"],
            "beats": [
                {
                    "id": "b1",
                    "time": "evening",
                    "place": "bedroom",
                    "place_detail": "a quiet, dim bedroom",
                    "event": "Had a hard, quiet evening at home",
                    "emotion": "subdued",
                    "people": [],
                    "objects": ["blanket", "lamp"],
                    "importance": 0.4,
                    "humor": 0.0,
                    "quote": None,
                }
            ],
            "people_mentioned": [],
            "flags": flags,
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


def _build_graph(tmp_path: Path) -> graph_module.DayGraph:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    return graph_module.DayGraph(
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        memory=InMemoryMemoryStore(),
    )


def _day_state(text: str) -> DayState:
    return DayState(
        job_id="job-daytype-1", user_id="u1", date=date(2026, 3, 1), source="text", text=text
    )


async def test_quiet_day_produces_a_valid_2_panel_strip(tmp_path: Path) -> None:
    provider, _ = routing.resolve("extractor")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("BeatSheet", _beat_sheet_json(flags=["too_short"]))
    script_provider, _ = routing.resolve("script")
    assert isinstance(script_provider, MockProvider)
    script_provider.set_structured_response(
        "Script",
        _two_panel_script_json(
            mood="quiet",
            captions=["Not much happened today.", "Just needed some rest."],
        ),
    )

    day_graph = _build_graph(tmp_path)
    result = await day_graph.run(_day_state("Quiet day, not much to say."))

    assert result.beats is not None
    assert "too_short" in result.beats.flags
    assert result.script is not None
    assert result.script.quiet_day is True
    assert len(result.script.panels) == 2

    assert result.layout is not None
    assert len(result.layout["strip"]["panels"]) == 2
    chosen = [c for c in result.candidates if c.chosen]
    assert {c.panel_id for c in chosen} == {1, 2}


async def test_sensitive_day_produces_a_valid_2_panel_strip_with_no_punchline_bubbles(
    tmp_path: Path,
) -> None:
    provider, _ = routing.resolve("extractor")
    assert isinstance(provider, MockProvider)
    provider.set_structured_response("BeatSheet", _beat_sheet_json(flags=["sensitive"]))
    script_provider, _ = routing.resolve("script")
    assert isinstance(script_provider, MockProvider)
    script_provider.set_structured_response(
        "Script",
        _two_panel_script_json(
            mood="subdued, gentle",
            captions=["It was a hard day.", "Glad it's over."],
        ),
    )

    day_graph = _build_graph(tmp_path)
    result = await day_graph.run(_day_state("Rough day, don't feel like elaborating."))

    assert result.beats is not None
    assert "sensitive" in result.beats.flags
    assert result.script is not None
    assert len(result.script.panels) == 2
    # A humor-suppressed day's script shouldn't carry a punchline bubble —
    # this only checks the FIXTURE reflects that (mocked), not that a
    # real LLM call chooses to omit one; see the module docstring.
    assert all(p.bubble == "" for p in result.script.panels)

    assert result.layout is not None
    assert len(result.layout["strip"]["panels"]) == 2
    chosen = [c for c in result.candidates if c.chosen]
    assert {c.panel_id for c in chosen} == {1, 2}
