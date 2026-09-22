"""OpenAI Chat Completions provider.

One of three files allowed to `import openai` (with `anthropic_provider.py`
and `mock_provider.py`).

Structured output uses `response_format={"type": "json_schema",
"json_schema": {"strict": True, ...}}` — `base.py`'s `to_strict_json_schema`
is an in-repo equivalent of the SDK's own strict-schema transform
(additionalProperties: false + every property required).
"""

from __future__ import annotations

import base64
import os
import time
from typing import Any

import openai

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
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)


class MissingAPIKeyError(RuntimeError):
    def __init__(self, env_var: str) -> None:
        super().__init__(f"{env_var} is not set (see .env.example)")


def _content_block(part: ContentPart) -> dict[str, Any]:
    if part.get("type") == "text":
        return {"type": "text", "text": part.get("text", "")}
    if url := part.get("url"):
        return {"type": "image_url", "image_url": {"url": url}}
    media_type, raw = image_part_to_data(part)
    data_uri = f"data:{media_type};base64,{base64.b64encode(raw).decode()}"
    return {"type": "image_url", "image_url": {"url": data_uri}}


def _to_openai_content(content: str | list[ContentPart]) -> str | list[dict[str, Any]]:
    if isinstance(content, str):
        return content
    return [_content_block(part) for part in content]


def _to_openai_messages(messages: list[Message]) -> list[dict[str, Any]]:
    # Unlike Anthropic, OpenAI takes "system" as a normal message role —
    # no splitting needed.
    return [{"role": m["role"], "content": _to_openai_content(m["content"])} for m in messages]


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None) -> None:
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise MissingAPIKeyError("OPENAI_API_KEY")
        self._client = openai.AsyncOpenAI(api_key=resolved_key)

    def _wrap_and_record(
        self,
        response: openai.types.chat.ChatCompletion,
        *,
        model: str,
        latency_ms: int,
        role: str | None,
    ) -> LLMResult:
        text = response.choices[0].message.content or ""
        usage = response.usage
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        usd = price_usd(self.name, model, input_tokens=input_tokens, output_tokens=output_tokens)
        result = LLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
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
        start = time.monotonic()
        response = await with_backoff(
            lambda: self._client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                # dict[str, Any] vs the SDK's ChatCompletionMessageParam
                # TypedDict union: structurally fine at runtime (the SDK
                # just serializes this to JSON), but mypy can't verify a
                # plain dict against a TypedDict union.
                messages=_to_openai_messages(messages),  # type: ignore[arg-type]
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
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema.__name__,
                "schema": to_strict_json_schema(schema, require_all=True),
                "strict": True,
            },
        }

        async def attempt(current_messages: list[Message]) -> tuple[str, LLMResult]:
            start = time.monotonic()
            response = await with_backoff(
                lambda: self._client.chat.completions.create(  # type: ignore[call-overload]
                    model=model,
                    temperature=temperature,
                    messages=_to_openai_messages(current_messages),
                    response_format=response_format,
                ),
                retryable=_RETRYABLE,
            )
            latency_ms = int((time.monotonic() - start) * 1000)
            result = self._wrap_and_record(response, model=model, latency_ms=latency_ms, role=None)
            return result.text, result

        return await structured_with_retry(schema, attempt, messages, retries=retries)
