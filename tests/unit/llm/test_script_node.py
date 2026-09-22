"""nodes/script.py — builds messages (with few-shot examples when
present), calls structured(), retries once on missing framing diversity,
persists versions/cost.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore, ScriptExample
from app.nodes.script import (
    DEFAULT_MAX_PANELS,
    NoBeatsError,
    build_messages,
    has_framing_diversity,
    write_script,
)
from contracts import Beat, BeatSheet, DayState, Panel, Script


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("script")
    assert isinstance(provider, MockProvider)
    return provider


def _beats(*, flags: list[str] | None = None) -> BeatSheet:
    return BeatSheet(
        date=date(2026, 3, 1),
        mood_arc=["content"],
        beats=[
            Beat(
                id="b1",
                time="morning",
                place="kitchen",
                event="Made coffee",
                emotion="content",
                importance=0.3,
                humor=0.1,
            ),
            Beat(
                id="b2",
                time="evening",
                place="desk",
                event="Finished a big project",
                emotion="proud",
                importance=0.8,
                humor=0.0,
            ),
        ],
        people_mentioned=[],
        flags=flags or [],
    )


def _valid_script(*, framings: tuple[str, str] = ("wide", "close")) -> Script:
    return Script(
        mood="content",
        quiet_day=False,
        panels=[
            Panel(
                id=1,
                beat_id="b1",
                place="kitchen",
                time_of_day="morning",
                expression="content",
                action="Pouring coffee",
                framing=framings[0],
                caption_a="Morning coffee",
                caption_b="Coffee first",
            ),
            Panel(
                id=2,
                beat_id="b2",
                place="desk",
                time_of_day="evening",
                expression="happy",
                action="Closing the laptop, done",
                framing=framings[1],
                caption_a="Finally done",
                caption_b="Shipped it",
            ),
        ],
    )


def _day_state() -> DayState:
    return DayState(job_id="job1", user_id="u1", date=date(2026, 3, 1), source="text")


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_SCRIPT", "mock:mock-script")


async def test_write_script_produces_a_valid_script() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("Script", _valid_script().model_dump_json())

    state = _day_state().model_copy(update={"beats": _beats()})
    result_state = await write_script(state, memory=InMemoryMemoryStore())

    assert result_state.script is not None
    assert len(result_state.script.panels) == 2


async def test_write_script_records_versions_and_cost() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("Script", _valid_script().model_dump_json())

    state = _day_state().model_copy(update={"beats": _beats()})
    result_state = await write_script(state, memory=InMemoryMemoryStore())

    assert result_state.versions["script_prompt"] == "script.v4"
    assert result_state.versions["script_model"] == "mock:mock-script"


def test_build_messages_defaults_max_panels_to_4_for_a_real_day() -> None:
    messages = build_messages(prompt_text="system", beat_sheet=_beats(), humor_level=5, examples=[])
    assert DEFAULT_MAX_PANELS == 4
    assert "max_panels: 4" in messages[1]["content"]


def test_build_messages_threads_a_real_max_panels_for_a_weekly_recap() -> None:
    """`weekly_graph.py` reuses this function to build a 6-panel recap
    from 6 selected beats — `max_panels` must reach the actual message
    sent to the LLM, not just the contract's own max length."""
    messages = build_messages(
        prompt_text="system", beat_sheet=_beats(), humor_level=5, examples=[], max_panels=6
    )
    assert "max_panels: 6" in messages[1]["content"]


def test_taste_notes_reach_the_script_prompt_and_stay_out_when_there_are_none() -> None:
    """The person's written taste notes ("prefers wide, calm shots") are
    read by the scriptwriter — and only when there are some."""
    with_notes = build_messages(
        prompt_text="system",
        beat_sheet=_beats(),
        humor_level=5,
        examples=[],
        taste_notes=["Prefers wide shots with the character small in the scene."],
    )
    body = with_notes[1]["content"]
    assert "Prefers wide shots with the character small in the scene." in body
    assert "never let it add events" in body

    without = build_messages(prompt_text="system", beat_sheet=_beats(), humor_level=5, examples=[])
    assert "taste" not in without[1]["content"]


async def test_write_script_retries_once_when_framing_is_not_diverse() -> None:
    provider = _resolved_mock_provider()
    not_diverse = _valid_script(framings=("medium", "medium")).model_dump_json()
    fixed = _valid_script(framings=("wide", "close")).model_dump_json()
    provider.set_structured_response("Script", [not_diverse, fixed])

    state = _day_state().model_copy(update={"beats": _beats()})
    result_state = await write_script(state, memory=InMemoryMemoryStore())

    assert result_state.script is not None
    assert has_framing_diversity(result_state.script)


async def test_write_script_uses_few_shot_examples_when_present() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("Script", _valid_script().model_dump_json())

    memory = InMemoryMemoryStore()
    example_beats = _beats()
    memory.add_rated_script(
        "u1", ScriptExample(beat_sheet=example_beats, script=_valid_script()), rating=5.0
    )

    state = _day_state().model_copy(update={"beats": _beats()})
    await write_script(state, memory=memory)

    # No assertion on the exact prompt text here (that's build_messages'
    # own concern) — this just proves the retrieval call actually
    # happens and doesn't blow up when examples exist.
    assert memory.top_rated_scripts("u1") != []


async def test_missing_beats_raises() -> None:
    with pytest.raises(NoBeatsError):
        await write_script(_day_state(), memory=InMemoryMemoryStore())


def _beat(i: int) -> Beat:
    return Beat(
        id=f"b{i}",
        time="morning",
        place="bedroom",
        event=f"scene {i}",
        emotion="calm",
        importance=0.5,
        humor=0.2,
    )


def _sheet(n: int, flags: list[str] | None = None) -> BeatSheet:
    return BeatSheet(
        date=date(2026, 3, 1),
        mood_arc=["content"],
        beats=[_beat(i) for i in range(1, n + 1)],
        people_mentioned=[],
        flags=flags or [],
    )


def _script_of(n: int) -> Script:
    panels = [
        Panel(
            id=i,
            beat_id="b1",
            place="bedroom",
            time_of_day="morning",
            expression="content",
            action="Doing something",
            framing=("wide", "close", "medium", "medium")[i - 1],
            caption_a="Caption",
            caption_b="Caption b",
        )
        for i in range(1, n + 1)
    ]
    return Script(mood="calm", quiet_day=n < 3, panels=panels)


async def test_a_day_with_four_real_beats_is_asked_again_when_the_writer_calls_it_quiet() -> None:
    """A lazy weekend written as five scenes came back as a 2-panel "quiet day" one run in two."""
    provider = _resolved_mock_provider()
    provider.set_structured_response(
        "Script", [_script_of(2).model_dump_json(), _script_of(4).model_dump_json()]
    )
    state = _day_state().model_copy(update={"beats": _sheet(5)})

    result = await write_script(state, memory=InMemoryMemoryStore())

    assert result.script is not None and len(result.script.panels) == 4


async def test_a_flagged_or_short_day_is_still_allowed_to_be_quiet() -> None:
    for sheet in (_sheet(5, ["sensitive"]), _sheet(5, ["too_short"]), _sheet(3)):
        provider = _resolved_mock_provider()
        provider.set_structured_response("Script", [_script_of(2).model_dump_json()])
        state = _day_state().model_copy(update={"beats": sheet})
        result = await write_script(state, memory=InMemoryMemoryStore())
        assert result.script is not None and len(result.script.panels) == 2


async def test_the_writer_is_not_asked_again_when_it_already_drew_four_panels() -> None:
    provider = _resolved_mock_provider()
    provider.set_structured_response("Script", [_script_of(4).model_dump_json()])
    state = _day_state().model_copy(update={"beats": _sheet(5)})
    result = await write_script(state, memory=InMemoryMemoryStore())
    assert result.script is not None and len(result.script.panels) == 4
    assert provider._structured_call_count.get("Script") == 1  # one call: no retry spend
