"""Canned-response provider — every unit test's only LLM provider.

Responses for `structured()` are keyed by `schema.__name__` and default to a
small built-in set that's always valid against the current contract, since
it's built from the model itself rather than hand-typed JSON that could
drift out of sync.

A registered value may be a list instead of a single string — each call
advances through it (repeating the last entry once exhausted), useful for
testing the retry-after-bad-output path.
"""

from __future__ import annotations

import time
from datetime import date

from app.llm.base import LLMResult, Message, T, structured_with_retry
from app.llm.cost import CostEvent, get_cost_sink
from app.llm.pricing import price_usd
from contracts import Beat, BeatSheet

_DEFAULT_BEAT_SHEET = BeatSheet(
    date=date(2026, 1, 1),
    mood_arc=["content"],
    beats=[
        Beat(
            id="b1",
            time="evening",
            place="kitchen",
            place_detail="small kitchen with a window",
            event="Cooked dinner and watched the sunset",
            emotion="content",
            objects=["pan", "window"],
            importance=0.6,
            humor=0.1,
        )
    ],
    people_mentioned=[],
).model_dump_json()

_DEFAULT_STRUCTURED_RESPONSES: dict[str, str | list[str]] = {
    "BeatSheet": _DEFAULT_BEAT_SHEET,
}


class MockProviderMissingResponseError(KeyError):
    def __init__(self, key: str) -> None:
        super().__init__(
            f"MockProvider has no canned response for {key!r}; pass "
            f"structured_responses={{{key!r}: '...'}} to its constructor"
        )


class MockProvider:
    name = "mock"

    def __init__(
        self,
        *,
        structured_responses: dict[str, str | list[str]] | None = None,
        complete_text: str = "This is a mock completion.",
    ) -> None:
        merged = {**_DEFAULT_STRUCTURED_RESPONSES, **(structured_responses or {})}
        self._structured_responses: dict[str, list[str]] = {
            key: ([value] if isinstance(value, str) else list(value))
            for key, value in merged.items()
        }
        self._structured_call_count: dict[str, int] = {}
        self.complete_text = complete_text

    def set_structured_response(self, key: str, value: str | list[str]) -> None:
        """Reconfigure a canned response after construction — e.g. when the
        provider instance came from `routing.resolve()` rather than being
        constructed directly, so there was no chance to pass it into
        `__init__`. `key` is the target schema's `__name__`, matching
        `structured()`'s own lookup."""
        self._structured_responses[key] = [value] if isinstance(value, str) else list(value)
        self._structured_call_count.pop(key, None)

    def _next_structured_raw(self, key: str) -> str:
        queue = self._structured_responses.get(key)
        if not queue:
            raise MockProviderMissingResponseError(key)
        index = self._structured_call_count.get(key, 0)
        self._structured_call_count[key] = index + 1
        return queue[min(index, len(queue) - 1)]

    def _record_and_wrap(self, text: str, model: str, role: str | None = None) -> LLMResult:
        # No real tokenizer for a canned string; chars/4 is a rough stand-in
        # good enough to exercise the cost-recording path.
        input_tokens, output_tokens = 0, max(1, len(text) // 4)
        usd = price_usd("mock", "mock", input_tokens=input_tokens, output_tokens=output_tokens)
        result = LLMResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            usd=usd,
            provider="mock",
            model=model,
            latency_ms=0,
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
        return self._record_and_wrap(self.complete_text, model)

    async def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str,
        temperature: float = 0,
        retries: int = 1,
    ) -> tuple[T, LLMResult]:
        key = schema.__name__

        async def attempt(_msgs: list[Message]) -> tuple[str, LLMResult]:
            raw = self._next_structured_raw(key)
            start = time.monotonic()
            result = self._record_and_wrap(raw, model)
            latency_ms = int((time.monotonic() - start) * 1000)
            return raw, result.model_copy(update={"latency_ms": latency_ms})

        return await structured_with_retry(schema, attempt, messages, retries=retries)
