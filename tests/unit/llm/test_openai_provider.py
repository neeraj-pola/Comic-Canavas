"""OpenAIProvider — a real recorded response replayed offline.

Same approach as `test_anthropic_provider.py`: `tests/cassettes/openai_extractor_001.json`
is one live `gpt-4o-mini` call against golden 001's transcript, frozen and
replayed by mocking only `chat.completions.create` — see that file's
docstring for why not vcrpy/HTTP-layer interception.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.llm.openai_provider import OpenAIProvider
from contracts import BeatSheet

CASSETTE = json.loads(
    (Path(__file__).parents[2] / "cassettes" / "openai_extractor_001.json").read_text()
)


def _fake_response() -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=CASSETTE["response_text"]))],
        usage=SimpleNamespace(
            prompt_tokens=CASSETTE["input_tokens"], completion_tokens=CASSETTE["output_tokens"]
        ),
    )


async def test_structured_returns_a_valid_beat_sheet_from_the_recorded_response() -> None:
    provider = OpenAIProvider(api_key="test-key-not-real")
    provider._client.chat.completions.create = AsyncMock(return_value=_fake_response())  # type: ignore[method-assign]

    sheet, result = await provider.structured(
        [
            {"role": "system", "content": "extract beats"},
            {"role": "user", "content": "date: 2026-02-02\ntranscript: ..."},
        ],
        BeatSheet,
        model=CASSETTE["model"],
    )

    assert isinstance(sheet, BeatSheet)
    assert len(sheet.beats) == 3
    assert "Priya" in sheet.people_mentioned
    assert result.provider == "openai"
    assert result.usd > 0
    assert result.input_tokens == CASSETTE["input_tokens"]


async def test_structured_request_uses_strict_json_schema_response_format() -> None:
    provider = OpenAIProvider(api_key="test-key-not-real")
    create = AsyncMock(return_value=_fake_response())
    provider._client.chat.completions.create = create  # type: ignore[method-assign]

    await provider.structured(
        [{"role": "user", "content": "extract"}], BeatSheet, model=CASSETTE["model"]
    )

    kwargs = create.call_args.kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["strict"] is True


def test_beat_sheet_and_anthropic_produce_the_same_field_set() -> None:
    """Gate 1: both providers must agree on the field set for the same fixture."""
    anthropic_cassette = json.loads(
        (Path(__file__).parents[2] / "cassettes" / "anthropic_extractor_001.json").read_text()
    )

    openai_sheet = BeatSheet.model_validate_json(CASSETTE["response_text"])
    anthropic_sheet = BeatSheet.model_validate_json(anthropic_cassette["response_text"])

    assert set(openai_sheet.model_dump().keys()) == set(anthropic_sheet.model_dump().keys())
    assert len(openai_sheet.beats) == len(anthropic_sheet.beats)
