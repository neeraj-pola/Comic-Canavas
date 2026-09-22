"""USD-per-1M-token pricing table.

Bump `PRICING_UPDATED` whenever a row below changes so a stale table is
easy to spot in review. Keyed by the same `"provider:model"` spec used in
`LLM_<ROLE>` env vars — `mock:mock` is free and exists so tests can still
exercise cost-recording code paths.
"""

from __future__ import annotations

from datetime import date

PRICING_UPDATED = date(2026, 9, 21)

# provider:model -> (input $/1M tokens, output $/1M tokens)
_PRICING_USD_PER_1M: dict[str, tuple[float, float]] = {
    "anthropic:claude-sonnet-4-6": (3.00, 15.00),
    "anthropic:claude-haiku-4-5-20251001": (1.00, 5.00),
    "openai:gpt-4o-mini": (0.15, 0.60),
    "openai:gpt-4o": (2.50, 10.00),
    "mock:mock": (0.0, 0.0),
}


class UnknownModelPricingError(ValueError):
    """Raised by `price_usd` for a `provider:model` with no pricing row.
    Deliberately fails loud rather than silently costing $0 — every
    external call's real spend must be logged accurately for the cost
    guard to work.
    """

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(
            f"no pricing for {provider}:{model} — add a row to "
            "services/worker/app/llm/pricing.py before using this model"
        )


def price_usd(provider: str, model: str, *, input_tokens: int, output_tokens: int) -> float:
    try:
        in_rate, out_rate = _PRICING_USD_PER_1M[f"{provider}:{model}"]
    except KeyError as exc:
        raise UnknownModelPricingError(provider, model) from exc
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000
