"""The provider-agnostic LLM interface.

Every node calls `llm.complete()` / `llm.structured()` through this module's
types — `anthropic_provider.py`, `openai_provider.py`, and `mock_provider.py`
are the only files allowed to import `anthropic`/`openai`. Switching provider
is `LLM_<ROLE>=<provider>:<model>` (see `routing.py`).
"""

from __future__ import annotations

import asyncio
import base64
import logging
from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("comiccanvas.llm")

T = TypeVar("T", bound=BaseModel)


class ContentPart(TypedDict, total=False):
    """One part of a multimodal message. `type: "text"` uses `text`;
    `type: "image"` uses exactly one of `url` (fetchable link) or `b64`
    (inline-encoded bytes) — both providers map either form to their own
    image-content shape.
    """

    type: Literal["text", "image"]
    text: str
    url: str
    b64: str


class Message(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class LLMResult(BaseModel):
    text: str
    input_tokens: int
    output_tokens: int
    usd: float
    provider: str
    model: str
    latency_ms: int


class StructuredOutputError(Exception):
    """Raised when `structured()` still fails schema validation after
    retrying with the validation error appended to the conversation.
    `raw_outputs` holds every raw attempt (in order) so the caller/logs can
    see exactly what the model returned each time.
    """

    def __init__(self, message: str, *, raw_outputs: list[str]) -> None:
        super().__init__(message)
        self.raw_outputs = raw_outputs


class LLMProvider(Protocol):
    name: str

    async def complete(
        self,
        messages: list[Message],
        *,
        model: str,
        temperature: float = 0,
        max_tokens: int = 1024,
    ) -> LLMResult: ...

    async def structured(
        self,
        messages: list[Message],
        schema: type[T],
        *,
        model: str,
        temperature: float = 0,
        retries: int = 1,
    ) -> tuple[T, LLMResult]: ...


# --- shared retry helper ----------------------------------------------------
#
# Each provider's structured() only needs to know how to make ONE attempt
# (its own tool-use / json_schema mechanics); this drives the "validate,
# and on failure re-ask once with the error appended" policy the same way
# for every provider, including the mock — so the retry behavior itself
# only needs testing once.

Attempt = Callable[[list[Message]], Awaitable[tuple[str, LLMResult]]]


async def structured_with_retry[R: BaseModel](
    schema: type[R],
    attempt: Attempt,
    messages: list[Message],
    *,
    retries: int = 1,
) -> tuple[R, LLMResult]:
    raw_outputs: list[str] = []
    current_messages = list(messages)

    for attempt_num in range(retries + 1):
        raw_text, result = await attempt(current_messages)
        raw_outputs.append(raw_text)
        try:
            parsed = schema.model_validate_json(raw_text)
        except ValidationError as exc:
            if attempt_num >= retries:
                raise StructuredOutputError(
                    f"{schema.__name__} failed validation after "
                    f"{attempt_num + 1} attempt(s): {exc}",
                    raw_outputs=raw_outputs,
                ) from exc
            current_messages = [
                *current_messages,
                {"role": "assistant", "content": raw_text},
                {
                    "role": "user",
                    "content": (
                        f"That response did not match the required schema:\n{exc}\n"
                        "Please correct it and return valid JSON only."
                    ),
                },
            ]
        else:
            return parsed, result

    raise AssertionError("unreachable: loop always returns or raises")


# --- shared backoff helper ---------------------------------------------------
#
# Each SDK raises its own exception types for 429/5xx, so the retryable set
# is supplied by the caller (each provider passes its own SDK's exception
# classes); the policy itself — 60s per-attempt timeout, 3 attempts,
# exponential backoff, logged — is identical across providers.


async def with_backoff[X](
    fn: Callable[[], Awaitable[X]],
    *,
    retryable: tuple[type[Exception], ...],
    max_attempts: int = 3,
    timeout_s: float = 60.0,
    base_delay_s: float = 1.0,
) -> X:
    last_exc: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return await asyncio.wait_for(fn(), timeout=timeout_s)
        except retryable as exc:
            last_exc = exc
            logger.warning("LLM call attempt %d/%d failed: %r", attempt, max_attempts, exc)
            if attempt < max_attempts:
                await asyncio.sleep(base_delay_s * (2 ** (attempt - 1)))
    assert last_exc is not None  # loop always sets it before falling through
    raise last_exc


# --- strict JSON schema ------------------------------------------------------
#
# Both providers' native structured-output modes only accept a constrained
# JSON Schema subset: every object needs `additionalProperties: false`;
# numeric/length bounds
# (`minimum`, `maxLength`, ...) aren't part of that subset and get moved
# into `description` as a best-effort hint instead of a hard constraint.
# Pydantic re-validates the actual response afterward (structured_with_retry
# above), so a value that slips past the hint still gets caught and retried
# — this only needs to be a good hint, not airtight. `require_all` mirrors
# OpenAI's strict mode, which (unlike Anthropic's) demands every property be
# in `required` (optionality is expressed by making the type nullable
# instead — which Pydantic's own `anyOf: [..., {"type": "null"}]` for
# `X | None` fields already does).
_UNSUPPORTED_KEYWORDS: dict[str, tuple[str, ...]] = {
    "string": ("minLength", "maxLength", "pattern"),
    "integer": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"),
    "number": ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"),
    "array": ("minItems", "maxItems"),
}


def _transform_schema_node(node: dict[str, Any], *, require_all: bool) -> dict[str, Any]:
    node = dict(node)
    if "$ref" in node:
        return node

    for key in ("anyOf", "oneOf", "allOf"):
        if key in node:
            node[key] = [_transform_schema_node(v, require_all=require_all) for v in node[key]]

    node_type = node.get("type")
    if node_type == "object" or "properties" in node:
        properties = node.get("properties", {})
        node["properties"] = {
            key: _transform_schema_node(value, require_all=require_all)
            for key, value in properties.items()
        }
        node["additionalProperties"] = False
        if require_all:
            node["required"] = list(properties.keys())
    elif node_type == "array" and "items" in node:
        node["items"] = _transform_schema_node(node["items"], require_all=require_all)

    leftover = {k: node.pop(k) for k in _UNSUPPORTED_KEYWORDS.get(node_type or "", ()) if k in node}
    if leftover:
        hint = ", ".join(f"{k}: {v}" for k, v in leftover.items())
        description = node.get("description", "")
        node["description"] = f"{description}\n\nConstraints: {{{hint}}}".strip()

    return node


def to_strict_json_schema(schema: type[BaseModel], *, require_all: bool = False) -> dict[str, Any]:
    raw = schema.model_json_schema()
    defs = raw.pop("$defs", None)
    result = _transform_schema_node(raw, require_all=require_all)
    if defs:
        result["$defs"] = {
            name: _transform_schema_node(node, require_all=require_all)
            for name, node in defs.items()
        }
    return result


def image_part_to_data(part: ContentPart) -> tuple[str, bytes]:
    """Decode a vision `ContentPart`'s `b64` into `(media_type, raw_bytes)`.
    Accepts a bare base64 string (assumed `image/jpeg`) or a full
    `data:image/png;base64,...` URI."""
    b64 = part.get("b64")
    if not b64:
        raise ValueError("image_part_to_data requires a ContentPart with 'b64' set")
    if b64.startswith("data:"):
        header, _, data = b64.partition(",")
        media_type = header.removeprefix("data:").split(";")[0] or "image/jpeg"
        return media_type, base64.b64decode(data)
    return "image/jpeg", base64.b64decode(b64)
