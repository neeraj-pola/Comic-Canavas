"""Anthropic Messages API provider.

One of three files allowed to `import anthropic` (with `openai_provider.py`
and `mock_provider.py`).

Structured output uses the Messages API's native `output_config.format`
(json_schema) rather than forced tool-use — it plugs into
`structured_with_retry` exactly as well, since both hand back plain
schema-shaped JSON text.

`temperature` is no longer a typed `messages.create()` parameter in this SDK
version for the newest models, so it's sent via `extra_body` instead, which
older models like the pinned default still honor.
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import anthropic

from app.llm.base import (
    ContentPart,
    LLMResult,
    Message,
    T,
    image_part_to_data,
    structured_with_retry,
    to_strict_json_schema,
    with_backoff,
)
from app.llm.cost import CostEvent, get_cost_sink
from app.llm.pricing import price_usd

_RETRYABLE: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.ServiceUnavailableError,
    anthropic.OverloadedError,
)

# structured() has no max_tokens in its protocol signature; BeatSheet/
# Script-shaped JSON comfortably fits well under this.
_STRUCTURED_MAX_TOKENS = 4096


class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_var: str) -> None:
        super().__init__(f"{env_var} is not set (see .env.example)")


def _drop_system(messages: list[Message]) -> list[Message]:
    return [m for m in messages if m["role"] != "system"]


def _system_text(messages: list[Message]) -> str | None:
    parts = [
        m["content"] for m in messages if m["role"] == "system" and isinstance(m["content"], str)
    ]
    return "\n\n".join(parts) if parts else None


def _content_block(part: ContentPart) -> dict[str, Any]:
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    if "url" in part:
        return {"type": "image", "source": {"type": "url", "url": part["url"]}}
    media_type, raw = image_part_to_data(part)
    data = base64.b64encode(raw).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def _to_anthropic_content(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_content_block(part) for part in content]


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    return [{"role": m["role"], "content": _to_anthropic_content(m["content"])} for m in messages]


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not resolved_key:
            raise MissingAPIKeyError("ANTHROPIC_API_KEY")
        self._client = anthropic.AsyncAnthropic(api_key=resolved_key)

    def _wrap_and_record(
        self, response: anthropic.types.Message, *, model: str, latency_ms: int, role: str | None
    ) -> LLMResult:
        text = "".join(block.text for block in response.content if block.type == "text")
        usd = price_usd(
            self.name,
            model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
        result = LLMResult(
            text=text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            usd=usd,
            provider=self.name,
            model=model,
            latency_ms=latency_ms,
        )
        get_cost_sink().record(
            CostEvent(
                provider=result.provider,
                model=result.model,
                role=role,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                usd=result.usd,
            )
        )
        return result

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResult:
        kwargs: dict[str, Any] = {"extra_body": {"temperature": temperature}}
        if system := _system_text(messages):
            kwargs["system"] = system

        start = time.monotonic()
        response = await with_backoff(
            lambda: self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                # dict[str, Any] vs the SDK's MessageParam TypedDict: see
                # base.py's Message/ContentPart docstrings.
                messages=_to_anthropic_messages(_drop_system(messages)),  # type: ignore[arg-type]
                **kwargs,
            ),
            retryable=_RETRYABLE,
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        return self._wrap_and_record(response, model=model, latency_ms=latency_ms, role=None)

    async def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str,
        temperature: float = 0,
        retries: int = 1,
    ) -> tuple[T, LLMResult]:
        system = _system_text(messages)
        output_format = {"type": "json_schema", "schema": to_strict_json_schema(schema)}

        async def attempt(current_messages: list[Message]) -> tuple[str, LLMResult]:
            kwargs: dict[str, Any] = {"extra_body": {"temperature": temperature}}
            if system:
                kwargs["system"] = system

            start = time.monotonic()
            response = await with_backoff(
                lambda: self._client.messages.create(  # type: ignore[call-overload]
                    model=model,
                    max_tokens=_STRUCTURED_MAX_TOKENS,
                    output_config={"format": output_format},
                    messages=_to_anthropic_messages(_drop_system(current_messages)),
                    **kwargs,
                ),
                retryable=_RETRYABLE,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            result = self._wrap_and_record(response, model=model, latency_ms=latency_ms, role=None)
            return result.text, result

        return await structured_with_retry(schema, attempt, messages, retries=retries)
