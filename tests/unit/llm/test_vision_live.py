"""Vision support, live proof frozen as a cassette.

`tests/cassettes/vision_color_check.json` was captured from real calls to
both providers with a locally-generated solid-blue PNG (as a `b64` data
URI `ContentPart`), asked "what color is this?" — both correctly answered
"Blue". Frozen and replayed here the same way as
`test_anthropic_provider.py`/`test_openai_provider.py`: mock only the SDK
client method, exercise the real `complete()` code path (including the
vision content-block mapping `test_vision_mapping.py` unit-tests in
isolation) against that real response.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from app.llm.anthropic_provider import AnthropicProvider
from app.llm.base import ContentPart
from app.llm.openai_provider import OpenAIProvider

CASSETTE = json.loads(
    (Path(__file__).parents[2] / "cassettes" / "vision_color_check.json").read_text()
)

# Not the real recorded image bytes (the cassette only kept the model's text
# response) — just needs to be *valid* base64, since the real content-block
# mapping (image_part_to_data) still runs client-side to build the request
# before the mocked SDK call short-circuits the actual network hop.
_PLACEHOLDER_B64 = base64.b64encode(b"not-the-real-image-bytes").decode()

_IMAGE_CONTENT: list[ContentPart] = [
    {"type": "image", "b64": f"data:image/png;base64,{_PLACEHOLDER_B64}"},
    {"type": "text", "text": "What single color is this image? Answer with one word."},
]


async def test_anthropic_vision_call_returns_the_recorded_answer() -> None:
    cassette = CASSETTE["anthropic"]
    provider = AnthropicProvider(api_key="test-key-not-real")
    provider._client.messages.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            content=[SimpleNamespace(type="text", text=cassette["response_text"])],
            usage=SimpleNamespace(
                input_tokens=cassette["input_tokens"], output_tokens=cassette["output_tokens"]
            ),
        )
    )

    result = await provider.complete(
        [{"role": "user", "content": _IMAGE_CONTENT}], model=cassette["model"], max_tokens=20
    )

    assert result.text == "Blue"
    assert result.usd > 0


async def test_openai_vision_call_returns_the_recorded_answer() -> None:
    cassette = CASSETTE["openai"]
    provider = OpenAIProvider(api_key="test-key-not-real")
    provider._client.chat.completions.create = AsyncMock(  # type: ignore[method-assign]
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=cassette["response_text"]))],
            usage=SimpleNamespace(
                prompt_tokens=cassette["input_tokens"], completion_tokens=cassette["output_tokens"]
            ),
        )
    )

    result = await provider.complete(
        [{"role": "user", "content": _IMAGE_CONTENT}], model=cassette["model"], max_tokens=20
    )

    assert result.text == "Blue"
    assert result.usd > 0
