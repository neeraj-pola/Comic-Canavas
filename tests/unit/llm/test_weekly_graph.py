"""The weekly recap graph — separate from the daily `DayGraph`. Every LLM
role, the image provider, and the four critic scoring functions are
mocked (same convention `test_graph.py` already established) — this test
is about the recap's own logic (top-6 beat selection, reuse-vs-generate
branching, composition), not re-verifying scoring accuracy or real LLM
behavior.
"""

from __future__ import annotations

import json
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from storage import LocalFileStorage

from app import weekly_graph
from app.images.base import IdentityRef
from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore
from app.nodes import score as score_module
from contracts import Beat, BeatSheet, Candidate, LookCard

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


def _beat(beat_id: str, event: str, *, importance: float) -> Beat:
    return Beat(
        id=beat_id,
        time="evening",
        place="kitchen",
        event=event,
        emotion="content",
        importance=importance,
        humor=0.2,
    )


def _week_beat_sheets() -> list[BeatSheet]:
    days = [
        (date(2026, 3, 2), "mon1", "Shipped the big feature at work", 0.9),
        (date(2026, 3, 3), "tue1", "Long walk with an old friend", 0.85),
        (date(2026, 3, 4), "wed1", "Cooked a new recipe that worked out", 0.8),
        (date(2026, 3, 5), "thu1", "Fixed the bug that had been nagging all week", 0.75),
        (date(2026, 3, 6), "fri1", "Ordinary quiet evening", 0.3),
        (date(2026, 3, 7), "sat1", "Weekend hike up the ridge trail", 0.6),
        (date(2026, 3, 8), "sun1", "Slow morning with coffee and a book", 0.5),
    ]
    return [
        BeatSheet(
            date=d,
            mood_arc=["steady"],
            beats=[_beat(beat_id, event, importance=importance)],
            people_mentioned=[],
        )
        for d, beat_id, event, importance in days
    ]


def _weekly_script_json() -> str:
    # `build_weekly_recap` renames every top beat to a synthetic,
    # guaranteed-unique "w{i}" id (by importance rank) before the script
    # LLM ever sees it, so a script's own `Panel.beat_id` values only ever
    # reference these synthetic ids — this mock's canned response must
    # match that, in the same importance order `_week_beat_sheets()`
    # produces: mon1(.9) -> tue1(.85) -> wed1(.8) -> thu1(.75) ->
    # sat1(.6) -> sun1(.5), with fri1(.3) dropped as the lowest of the 7.
    beat_order = ["w0", "w1", "w2", "w3", "w4", "w5"]
    framings = ["wide", "close", "medium", "medium", "wide", "close"]
    panels = [
        {
            "id": i,
            "beat_id": beat_id,
            "place": "kitchen",
            "time_of_day": "evening",
            "expression": "content",
            "action": f"scene for {beat_id}",
            "framing": framing,
            "caption_a": f"Caption for {beat_id}.",
            "caption_b": "",
            "bubble": "",
            "cast": [],
        }
        for i, (beat_id, framing) in enumerate(zip(beat_order, framings, strict=True), start=1)
    ]
    return json.dumps({"mood": "a good week overall", "quiet_day": False, "panels": panels})


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
    provider.set_structured_response("Script", _weekly_script_json())
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


def _seed_reused_panels(
    memory: InMemoryMemoryStore, beat_sheets: list[BeatSheet], *, storage: LocalFileStorage
) -> None:
    """Simulates 4 of the week's beats already having a chosen daily
    panel (`write_beats(..., panel_urls=...)`) — only "sat1"/"sun1" are
    left without one, so the recap must generate exactly those 2 fresh.
    Writes a real (tiny, solid-color) placeholder image into `storage`
    for each reused beat — `compose_strip` needs real, fetchable bytes at
    that URL, not a fabricated one `storage.get_object_by_url` would
    correctly reject."""
    reused = {"mon1", "tue1", "wed1", "thu1"}
    placeholder = Image.new("RGB", (64, 64), (120, 140, 200))
    buf = BytesIO()
    placeholder.save(buf, format="PNG")
    placeholder_bytes = buf.getvalue()

    for sheet in beat_sheets:
        beat = sheet.beats[0]
        panel_urls = None
        if beat.id in reused:
            url = storage.put_object(
                f"daily-reuse/{beat.id}.png", placeholder_bytes, content_type="image/png"
            )
            panel_urls = {beat.id: [url]}
        memory.write_beats("u1", sheet, panel_urls=panel_urls)


async def test_weekly_recap_generates_at_most_2_new_panels(tmp_path: Path) -> None:
    """No more than 2 new panels generated when 5 of 7 beats already have a chosen daily panel."""
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    beat_sheets = _week_beat_sheets()
    _seed_reused_panels(memory, beat_sheets, storage=storage)

    recap = await weekly_graph.build_weekly_recap(
        "u1",
        beat_sheets,
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        iso_week="2026-W10",
        memory=memory,
    )

    assert recap.new_generations <= 2
    assert recap.new_generations == 2  # exactly sat1 and sun1 in this fixture


async def test_weekly_recap_selects_the_top_6_beats_by_importance(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    beat_sheets = _week_beat_sheets()
    _seed_reused_panels(memory, beat_sheets, storage=storage)

    recap = await weekly_graph.build_weekly_recap(
        "u1",
        beat_sheets,
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        iso_week="2026-W10",
        memory=memory,
    )

    selected_beat_ids = {p.beat_id for p in recap.script.panels}
    # Real ids are "w0".."w5" now (see `_weekly_script_json`'s own
    # docstring) — six unique panels is itself proof the lowest-
    # importance beat (fri1, 0.3) was correctly dropped from the 7.
    assert selected_beat_ids == {"w0", "w1", "w2", "w3", "w4", "w5"}


async def test_weekly_recap_reuses_daily_panel_urls_directly(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    beat_sheets = _week_beat_sheets()
    _seed_reused_panels(memory, beat_sheets, storage=storage)

    recap = await weekly_graph.build_weekly_recap(
        "u1",
        beat_sheets,
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        iso_week="2026-W10",
        memory=memory,
    )

    # `_seed_reused_panels` marks the ORIGINAL "mon1"/"tue1"/"wed1"/
    # "thu1" as already having a daily panel; those map to the new
    # synthetic "w0".."w3" (see `_weekly_script_json`'s own docstring
    # for the importance-order mapping) — "w4"/"w5" (sat1/sun1) don't,
    # so those 2 must be freshly generated.
    original_reused_ids = {"mon1", "tue1", "wed1", "thu1"}
    expected_urls = {
        storage.get_url(f"daily-reuse/{beat_id}.png") for beat_id in original_reused_ids
    }
    reused_beat_ids = {"w0", "w1", "w2", "w3"}
    generated_beat_ids = {"w4", "w5"}

    reused_candidates = [c for c in recap.candidates if c.id.startswith("reused-")]
    assert {c.url for c in reused_candidates} == expected_urls
    assert {c.id.rsplit("-", 1)[-1] for c in reused_candidates} == reused_beat_ids

    generated_candidates = [c for c in recap.candidates if not c.id.startswith("reused-")]
    assert len(generated_candidates) == len(generated_beat_ids)


async def test_weekly_recap_disambiguates_colliding_beat_ids_across_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`Beat.id` is only unique within one day (the extractor restarts "b1", "b2" every day), and
    two real days both extracting their own most-important beat as "b1" is a plausible, common
    case — each must resolve to its own distinct stored panel, not collide in `get_panel_urls`."""
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    beat_sheets = [
        BeatSheet(
            date=date(2026, 3, 2),
            mood_arc=["steady"],
            beats=[_beat("b1", "Shipped the big feature at work", importance=0.9)],
            people_mentioned=[],
        ),
        BeatSheet(
            date=date(2026, 3, 3),
            mood_arc=["steady"],
            beats=[_beat("b1", "Long walk with an old friend", importance=0.85)],
            people_mentioned=[],
        ),
    ]

    placeholder_a = BytesIO()
    Image.new("RGB", (64, 64), (120, 140, 200)).save(placeholder_a, format="PNG")
    placeholder_b = BytesIO()
    Image.new("RGB", (64, 64), (200, 140, 120)).save(placeholder_b, format="PNG")
    url_day1 = storage.put_object(
        "daily-reuse/day1-b1.png", placeholder_a.getvalue(), content_type="image/png"
    )
    url_day2 = storage.put_object(
        "daily-reuse/day2-b1.png", placeholder_b.getvalue(), content_type="image/png"
    )
    memory.write_beats("u1", beat_sheets[0], panel_urls={"b1": [url_day1]})
    memory.write_beats("u1", beat_sheets[1], panel_urls={"b1": [url_day2]})

    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_EXTRACTOR", "mock:mock-extractor")
    monkeypatch.setenv("LLM_SCRIPT", "mock:mock-script")
    monkeypatch.setenv("LLM_PROMPTS", "mock:mock-prompts")
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    # Rank order by importance: day1 (0.9) -> "w0", day2 (0.85) -> "w1".
    panels = [
        {
            "id": i,
            "beat_id": beat_id,
            "place": "kitchen",
            "time_of_day": "evening",
            "expression": "content",
            "action": f"scene for {beat_id}",
            "framing": framing,
            "caption_a": f"Caption for {beat_id}.",
            "caption_b": "",
            "bubble": "",
            "cast": [],
        }
        for i, (beat_id, framing) in enumerate([("w0", "wide"), ("w1", "close")], start=1)
    ]
    provider.set_structured_response(
        "Script", json.dumps({"mood": "a good week", "quiet_day": False, "panels": panels})
    )
    provider.set_structured_response("PanelClauses", _clauses_json())

    recap = await weekly_graph.build_weekly_recap(
        "u1",
        beat_sheets,
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        iso_week="2026-W10",
        memory=memory,
    )

    reused_by_panel = {c.panel_id: c.url for c in recap.candidates if c.id.startswith("reused-")}
    assert reused_by_panel[1] == url_day1  # w0 (day1's "b1") resolves to day1's own image
    assert reused_by_panel[2] == url_day2  # w1 (day2's "b1") resolves to day2's own image
    assert recap.new_generations == 0  # both reused correctly, nothing generated


async def test_weekly_recap_produces_a_valid_6_panel_strip(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")
    memory = InMemoryMemoryStore()
    beat_sheets = _week_beat_sheets()
    _seed_reused_panels(memory, beat_sheets, storage=storage)

    recap = await weekly_graph.build_weekly_recap(
        "u1",
        beat_sheets,
        storage=storage,
        identity=IdentityRef(trigger_token="sks person"),
        look_card=_look_card(),
        iso_week="2026-W10",
        memory=memory,
    )

    assert recap.strip_url is not None
    assert recap.story_url is not None
    assert len(recap.script.panels) == 6
    assert len(recap.layout["strip"]["panels"]) == 6
    assert len(recap.candidates) == 6


async def test_weekly_recap_raises_on_an_empty_week(tmp_path: Path) -> None:
    storage = LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")

    with pytest.raises(weekly_graph.NoBeatsInWeekError):
        await weekly_graph.build_weekly_recap(
            "u1",
            [],
            storage=storage,
            identity=IdentityRef(trigger_token="sks person"),
            look_card=_look_card(),
            iso_week="2026-W10",
            memory=InMemoryMemoryStore(),
        )
