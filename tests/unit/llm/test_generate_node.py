"""Tasks 4.5/4.6/4.7: nodes/generate.py — fans generation out across all
panels concurrently, checks the cost guard first, and records a cost
event per panel.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from storage import LocalFileStorage

from app.images import routing
from app.images.base import IdentityRef, ImageProvider
from app.llm.cost import InMemoryCostEventSink, get_cost_sink, set_cost_sink
from app.nodes.generate import (
    CostGuardExceededError,
    NoPromptsError,
    generate_panels,
)
from contracts import Candidate, DayState, ImagePrompt


def _storage(tmp_path: Path) -> LocalFileStorage:
    return LocalFileStorage(root=tmp_path, base_url="http://localhost:8000")


def _identity() -> IdentityRef:
    return IdentityRef(trigger_token="sks person")


def _prompt(panel_id: int) -> ImagePrompt:
    return ImagePrompt(
        panel_id=panel_id,
        generator="mock",
        character_clause="[IDENTITY], content expression",
        environment_clause="a chipped mug, morning light, a wooden table, steam",
        positive=f"[IDENTITY], content expression, panel {panel_id}",
        negative="no artist names",
        seed=100 + panel_id,
    )


def _day_state(*, n_panels: int = 4) -> DayState:
    return DayState(
        job_id="job-1",
        user_id="u1",
        date=date(2026, 3, 1),
        source="text",
        prompts=[_prompt(i) for i in range(1, n_panels + 1)],
    )


class _FakeExpensiveProvider:
    """A stand-in `ImageProvider` whose estimate_usd isn't free, so the
    cost guard has something real to trip on — MockImageProvider itself
    always estimates $0, so it can never exercise that path."""

    name = "fake-expensive"

    async def generate(
        self, prompt: ImagePrompt, *, n: int, identity: IdentityRef
    ) -> list[Candidate]:
        return [
            Candidate(id=f"{prompt.panel_id}-{i}", panel_id=prompt.panel_id, url="http://x", seed=i)
            for i in range(n)
        ]

    def estimate_usd(self, *, n: int) -> float:
        return 1.0 * n


@pytest.fixture(autouse=True)
def _cost_sink() -> Iterator[None]:
    sink = InMemoryCostEventSink()
    previous = get_cost_sink()
    set_cost_sink(sink)
    yield
    set_cost_sink(previous)


async def test_generate_panels_produces_n_times_panels_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    state = _day_state(n_panels=4)

    result_state = await generate_panels(
        state, storage=_storage(tmp_path), identity=_identity(), n=3
    )

    assert len(result_state.candidates) == 12
    assert sorted({c.panel_id for c in result_state.candidates}) == [1, 2, 3, 4]


async def test_generate_panels_is_fast_under_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    state = _day_state(n_panels=4)

    start = time.monotonic()
    result_state = await generate_panels(
        state, storage=_storage(tmp_path), identity=_identity(), n=3
    )
    elapsed = time.monotonic() - start

    assert len(result_state.candidates) == 12
    assert elapsed < 2.0


async def test_generate_panels_raises_when_prompts_empty(tmp_path: Path) -> None:
    with pytest.raises(NoPromptsError):
        await generate_panels(
            _day_state(n_panels=0), storage=_storage(tmp_path), identity=_identity()
        )


async def test_generate_panels_records_a_cost_event_per_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("IMAGE_PROVIDER", "mock")
    state = _day_state(n_panels=2)

    await generate_panels(state, storage=_storage(tmp_path), identity=_identity(), n=3)

    events = get_cost_sink().events  # type: ignore[attr-defined]
    assert len(events) == 2
    assert all(e.provider == "mock" and e.role == "image" and e.units == 3 for e in events)


async def test_cost_guard_aborts_before_generating_when_projection_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake: ImageProvider = _FakeExpensiveProvider()
    monkeypatch.setattr(routing, "resolve", lambda storage: fake)
    monkeypatch.setenv("MAX_USD_PER_DAY", "0.10")
    state = _day_state(n_panels=4)

    with pytest.raises(CostGuardExceededError, match=r"\$4\.00.*\$0\.10"):
        await generate_panels(state, storage=_storage(tmp_path), identity=_identity(), n=1)

    assert get_cost_sink().events == []  # type: ignore[attr-defined]
