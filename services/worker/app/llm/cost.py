"""`cost_events` recording — provider-agnostic, DB-agnostic.

Providers record through the `CostEventSink` protocol below, so swapping
`InMemoryCostEventSink` (the default, and what tests use) for a
Postgres-backed one changes nothing else — same shape as
`packages/storage`'s Local/R2 split.

Not just LLM calls: every external call (LLM, ASR, image) should log one of
these — `images/*.py` records through the same sink with
`input_tokens`/`output_tokens` left at 0 and `units` set to the image count
instead.
"""

from __future__ import annotations

from typing import Protocol

from pydantic import BaseModel


class CostEvent(BaseModel):
    provider: str
    model: str
    role: str | None = None  # extractor/script/prompts/judge/lookcard — see routing.py
    input_tokens: int = 0
    output_tokens: int = 0
    units: int = 0  # non-token calls: images generated, audio seconds transcribed, ...
    usd: float


class CostEventSink(Protocol):
    def record(self, event: CostEvent) -> None: ...


class InMemoryCostEventSink:
    """Default sink; also what tests assert against until Phase 8."""

    def __init__(self) -> None:
        self.events: list[CostEvent] = []

    def record(self, event: CostEvent) -> None:
        self.events.append(event)


_sink: CostEventSink = InMemoryCostEventSink()


def get_cost_sink() -> CostEventSink:
    return _sink


def set_cost_sink(sink: CostEventSink) -> None:
    global _sink
    _sink = sink
