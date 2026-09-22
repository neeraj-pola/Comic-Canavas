"""Structured-output retry — one re-ask with the error appended, then
`StructuredOutputError` with both raw attempts attached.
"""

from __future__ import annotations

import pytest

from app.llm.base import Message, StructuredOutputError
from app.llm.mock_provider import MockProvider
from contracts import Beat, BeatSheet

_VALID = BeatSheet(
    date="2026-01-01",
    mood_arc=["content"],
    beats=[
        Beat(
            id="b1",
            time="evening",
            place="kitchen",
            event="Cooked dinner",
            emotion="content",
            importance=0.5,
            humor=0.1,
        )
    ],
).model_dump_json()


async def test_bad_json_then_good_json_succeeds_on_retry() -> None:
    provider = MockProvider(structured_responses={"BeatSheet": ["not valid json at all", _VALID]})
    sheet, result = await provider.structured(
        [{"role": "user", "content": "extract"}], BeatSheet, model="m", retries=1
    )
    assert sheet.beats[0].event == "Cooked dinner"
    assert result.text == _VALID  # the LLMResult returned is from the *successful* attempt


async def test_bad_json_twice_raises_with_both_raw_outputs() -> None:
    provider = MockProvider(structured_responses={"BeatSheet": ["bad one", "bad two"]})
    messages: list[Message] = [{"role": "user", "content": "extract"}]
    with pytest.raises(StructuredOutputError) as exc_info:
        await provider.structured(messages, BeatSheet, model="m", retries=1)
    assert exc_info.value.raw_outputs == ["bad one", "bad two"]


async def test_retries_zero_means_a_single_attempt() -> None:
    provider = MockProvider(structured_responses={"BeatSheet": ["bad", _VALID]})
    messages: list[Message] = [{"role": "user", "content": "extract"}]
    with pytest.raises(StructuredOutputError) as exc_info:
        await provider.structured(messages, BeatSheet, model="m", retries=0)
    assert exc_info.value.raw_outputs == ["bad"]
