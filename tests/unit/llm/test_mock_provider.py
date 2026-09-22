"""MockProvider — canned responses, zero network calls.

`make test`'s zero-network-calls guarantee is enforced globally by
pytest-socket (`--disable-socket` in the root pyproject.toml); this file
exercises the provider's actual behavior, not just its lack of sockets.
"""

from __future__ import annotations

import pytest

from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from app.llm.mock_provider import MockProvider, MockProviderMissingResponseError
from contracts import Beat, BeatSheet, Panel


async def test_complete_returns_configured_text() -> None:
    provider = MockProvider(complete_text="hello from the mock")
    result = await provider.complete([{"role": "user", "content": "hi"}], model="mock-model")
    assert result.text == "hello from the mock"
    assert result.provider == "mock"
    assert result.model == "mock-model"


async def test_structured_returns_default_beat_sheet() -> None:
    provider = MockProvider()
    sheet, result = await provider.structured(
        [{"role": "user", "content": "extract beats"}], BeatSheet, model="mock-model"
    )
    assert isinstance(sheet, BeatSheet)
    assert len(sheet.beats) >= 1
    assert result.provider == "mock"


async def test_structured_response_is_overridable() -> None:
    custom = BeatSheet(
        date="2026-01-02",
        mood_arc=["tired"],
        beats=[
            Beat(
                id="b1",
                time="morning",
                place="bedroom",
                event="Woke up late",
                emotion="tired",
                importance=0.3,
                humor=0.0,
            )
        ],
    )
    provider = MockProvider(structured_responses={"BeatSheet": custom.model_dump_json()})
    sheet, _ = await provider.structured([{"role": "user", "content": "x"}], BeatSheet, model="m")
    assert sheet.mood_arc == ["tired"]
    assert sheet.beats[0].event == "Woke up late"


async def test_missing_structured_response_raises_clear_error() -> None:
    # Panel isn't in MockProvider's built-in defaults (only BeatSheet is) and
    # this test doesn't register one — the missing-response error should
    # surface through the public structured() call, not just the private
    # lookup, since that's what a node actually calling it would hit.
    provider = MockProvider()
    with pytest.raises(MockProviderMissingResponseError, match="Panel"):
        await provider.structured([{"role": "user", "content": "x"}], Panel, model="m")


async def test_calls_record_a_cost_event() -> None:
    sink = InMemoryCostEventSink()
    previous = get_cost_sink()
    set_cost_sink(sink)
    try:
        provider = MockProvider()
        await provider.complete([{"role": "user", "content": "hi"}], model="mock-model")
    finally:
        set_cost_sink(previous)
    assert len(sink.events) == 1
    assert sink.events[0].provider == "mock"
