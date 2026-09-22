"""AnthropicProvider — a real recorded response replayed offline.

`tests/cassettes/anthropic_extractor_001.json` was captured from one live
`claude-sonnet-4-6` call against golden 001's fixture transcript, then
frozen — `make test` never calls the network
(pytest-socket) to reach it. This mocks only the SDK client method
(`messages.create`), not the HTTP transport, so it exercises the real
`structured()` code path (message building, the strict json_schema
request, response parsing, cost calculation) end to end against that
real response — a deliberate choice over vcrpy/HTTP-layer interception,
since Anthropic's current SDK is built on `httpx2` (see
anthropic_provider.py's docstring on SDK drift), which off-the-shelf VCR
tooling doesn't know how to patch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.llm.anthropic_provider import AnthropicProvider
from contracts import BeatSheet

CASSETTE = json.loads(
    (Path(__file__).parents[2] / "cassettes" / "anthropic_extractor_001.json").read_text()
)


def _fake_response() -> SimpleNamespace:
    return SimpleNamespace(
        content=[SimpleNamespace(type="text", text=CASSETTE["response_text"])],
        usage=SimpleNamespace(
            input_tokens=CASSETTE["input_tokens"], output_tokens=CASSETTE["output_tokens"]
        ),
    )


async def test_structured_returns_a_valid_beat_sheet_from_the_recorded_response() -> None:
    provider = AnthropicProvider(api_key="test-key-not-real")
    provider._client.messages.create = AsyncMock(return_value=_fake_response())  # type: ignore[method-assign]

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
    assert result.provider == "anthropic"
    assert result.usd > 0
    assert result.input_tokens == CASSETTE["input_tokens"]


async def test_structured_request_uses_native_json_schema_format() -> None:
    provider = AnthropicProvider(api_key="test-key-not-real")
    create = AsyncMock(return_value=_fake_response())
    provider._client.messages.create = create  # type: ignore[method-assign]

    await provider.structured(
        [{"role": "user", "content": "extract"}], BeatSheet, model=CASSETTE["model"]
    )

    kwargs = create.call_args.kwargs
    assert kwargs["output_config"]["format"]["type"] == "json_schema"
    assert kwargs["extra_body"] == {"temperature": 0}
