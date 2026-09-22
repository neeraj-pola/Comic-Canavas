"""Pricing table + cost_events recording.

`make test` never calls a real provider (pytest-socket), so that calling
structured() inserts a row with usd > 0 is proven in two
parts here: `price_usd` is a pure function, directly testable for a real
priced model; and the record-after-price pattern every provider's
`_wrap_and_record` follows is exercised directly with that same real
model's numbers, showing a nonzero-usd row reaches the sink exactly the
way a real `structured()` call's result would. `test_mock_provider.py`
separately proves the same call sites fire for the mock provider (usd==0
there, correctly — mock pricing is free).
"""

from __future__ import annotations

import pytest

from app.llm.cost import CostEvent, InMemoryCostEventSink, get_cost_sink, set_cost_sink
from app.llm.pricing import UnknownModelPricingError, price_usd


def test_price_usd_is_positive_for_a_real_model() -> None:
    usd = price_usd("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=500)
    assert usd > 0


def test_price_usd_scales_with_tokens() -> None:
    small = price_usd("openai", "gpt-4o-mini", input_tokens=100, output_tokens=100)
    large = price_usd("openai", "gpt-4o-mini", input_tokens=1000, output_tokens=1000)
    assert large == pytest.approx(small * 10)


def test_price_usd_zero_for_mock() -> None:
    assert price_usd("mock", "mock", input_tokens=1000, output_tokens=1000) == 0.0


def test_unknown_model_raises_instead_of_costing_zero() -> None:
    with pytest.raises(UnknownModelPricingError, match="anthropic:claude-nonexistent"):
        price_usd("anthropic", "claude-nonexistent", input_tokens=10, output_tokens=10)


def test_recording_a_real_priced_call_yields_a_positive_usd_row() -> None:
    sink = InMemoryCostEventSink()
    previous = get_cost_sink()
    set_cost_sink(sink)
    try:
        usd = price_usd("anthropic", "claude-sonnet-4-6", input_tokens=2000, output_tokens=400)
        get_cost_sink().record(
            CostEvent(
                provider="anthropic",
                model="claude-sonnet-4-6",
                role="script",
                input_tokens=2000,
                output_tokens=400,
                usd=usd,
            )
        )
    finally:
        set_cost_sink(previous)

    assert len(sink.events) == 1
    assert sink.events[0].usd > 0
    assert sink.events[0].role == "script"
