"""nodes/beats.py — builds messages, calls structured(), enforces
`too_short`, persists via the memory store.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.llm import routing
from app.llm.mock_provider import MockProvider
from app.memory import InMemoryMemoryStore
from app.nodes.beats import NoTranscriptError, extract_beats
from contracts import Beat, BeatSheet, DayState


def _resolved_mock_provider() -> MockProvider:
    provider, _ = routing.resolve("extractor")
    assert isinstance(provider, MockProvider)
    return provider


def _day_state(transcript: str | None) -> DayState:
    return DayState(
        job_id="job1", user_id="u1", date=date(2026, 3, 4), source="text", transcript=transcript
    )


_FIXTURE_TRANSCRIPT = (
    "Okay so today. Woke up late because I forgot to set an alarm, total panic "
    "getting out the door. Then at the office we had the all-hands and I found "
    "out the launch got pushed a week, which honestly was a relief because I "
    "was not ready. Grabbed lunch with Sam at that noodle place near the "
    "office, first time catching up properly in months. And then in the "
    "evening I just collapsed on the couch and watched something dumb, "
    "completely wiped out."
)

_FOUR_BEATS = BeatSheet(
    date=date(2026, 3, 4),
    mood_arc=["panicked", "relieved", "content", "tired"],
    beats=[
        Beat(
            id="b1",
            time="morning",
            place="bedroom",
            place_detail="a bedroom with an alarm clock that didn't go off",
            event="Woke up late after the alarm didn't go off and rushed out the door",
            emotion="panicked",
            importance=0.3,
            humor=0.4,
            quote="total panic getting out the door",
        ),
        Beat(
            id="b2",
            time="midday",
            place="desk",
            place_detail="an office all-hands meeting room",
            event="Learned at the all-hands that the launch was pushed a week",
            emotion="relieved",
            importance=0.7,
            humor=0.0,
            quote="which honestly was a relief because I was not ready",
        ),
        Beat(
            id="b3",
            time="midday",
            place="cafe",
            place_detail="a noodle place near the office",
            event="Had lunch with Sam and caught up properly for the first time in months",
            emotion="content",
            people=["Samantha"],
            importance=0.5,
            humor=0.1,
            quote="first time catching up properly in months",
        ),
        Beat(
            id="b4",
            time="evening",
            place="other",
            place_detail="a living room couch in front of the TV",
            event="Collapsed on the couch and watched something dumb",
            emotion="tired",
            importance=0.2,
            humor=0.2,
            quote="completely wiped out",
        ),
    ],
    people_mentioned=["Samantha"],
    flags=[],
)


@pytest.fixture(autouse=True)
def _mock_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    routing.reset_provider_cache()
    monkeypatch.setenv("LLM_EXTRACTOR", "mock:mock-extractor")


async def test_extract_beats_yields_the_four_expected_beats() -> None:
    memory = InMemoryMemoryStore()
    memory.seed("u1", people=["Samantha"])
    provider = _resolved_mock_provider()
    provider.set_structured_response("BeatSheet", _FOUR_BEATS.model_dump_json())

    state = _day_state(_FIXTURE_TRANSCRIPT)
    result_state = await extract_beats(state, memory=memory)

    assert result_state.beats is not None
    assert len(result_state.beats.beats) == 4
    assert [b.event for b in result_state.beats.beats] == [b.event for b in _FOUR_BEATS.beats]


async def test_extract_beats_records_versions_and_cost() -> None:
    memory = InMemoryMemoryStore()
    provider = _resolved_mock_provider()
    provider.set_structured_response("BeatSheet", _FOUR_BEATS.model_dump_json())

    state = _day_state(_FIXTURE_TRANSCRIPT)
    result_state = await extract_beats(state, memory=memory)

    assert result_state.versions["extractor_prompt"] == "extractor.v1"
    assert result_state.versions["extractor_model"] == "mock:mock-extractor"
    assert result_state.cost_usd == state.cost_usd  # mock pricing is $0


async def test_extract_beats_persists_via_memory_store() -> None:
    memory = InMemoryMemoryStore()
    provider = _resolved_mock_provider()
    provider.set_structured_response("BeatSheet", _FOUR_BEATS.model_dump_json())

    state = _day_state(_FIXTURE_TRANSCRIPT)
    await extract_beats(state, memory=memory)

    assert "Samantha" in memory.known_people("u1")


async def test_fewer_than_three_beats_gets_too_short_flag() -> None:
    memory = InMemoryMemoryStore()
    thin_sheet = BeatSheet(
        date=date(2026, 3, 4),
        mood_arc=["flat"],
        beats=[_FOUR_BEATS.beats[0]],
        people_mentioned=[],
    )
    provider = _resolved_mock_provider()
    provider.set_structured_response("BeatSheet", thin_sheet.model_dump_json())

    state = _day_state("Not much happened.")
    result_state = await extract_beats(state, memory=memory)

    assert result_state.beats is not None
    assert "too_short" in result_state.beats.flags


async def test_missing_transcript_raises() -> None:
    state = _day_state(None)
    with pytest.raises(NoTranscriptError):
        await extract_beats(state, memory=InMemoryMemoryStore())
