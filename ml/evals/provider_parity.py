"""Gate 1: run the extractor fixture on both providers, confirm they
return the same `BeatSheet` field set, print total spend (must stay
under $0.05). `make test-live` — needs real `ANTHROPIC_API_KEY` and
`OPENAI_API_KEY`; unlike `ml/evals/extractor.py`'s pure half, nothing
here is exercised by `make test`.

Run via `uv run python -m ml.evals.provider_parity` (from the repo root)
— not a plain script path like ml/evals/extractor.py's docstring
describes: this file also imports its sibling `ml.evals.extractor`,
which needs `ml` itself resolvable as a package (repo root on sys.path,
which only `-m` from the repo root gives it — see tests/unit/ml's
pattern in scripts/test-all.sh for the same requirement). It still
bootstraps services/worker onto sys.path itself below, for app.*.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_WORKER_APP_ROOT = _REPO_ROOT / "services" / "worker"
if str(_WORKER_APP_ROOT) not in sys.path:
    sys.path.insert(0, str(_WORKER_APP_ROOT))

from app.llm.anthropic_provider import AnthropicProvider  # noqa: E402
from app.llm.openai_provider import OpenAIProvider  # noqa: E402
from app.nodes.beats import build_messages  # noqa: E402
from app.prompts.loader import load_prompt  # noqa: E402
from contracts import BeatSheet  # noqa: E402
from ml.evals.extractor import GOLDENS_DIR, load_goldens  # noqa: E402

FIXTURE_ID = "001_real_commute_and_small_win"
SPEND_CAP_USD = 0.05

PROVIDERS: list[tuple[str, type[AnthropicProvider] | type[OpenAIProvider], str]] = [
    ("anthropic", AnthropicProvider, "claude-sonnet-4-6"),
    ("openai", OpenAIProvider, "gpt-4o-mini"),
]


async def main() -> None:
    golden = next(g for g in load_goldens(GOLDENS_DIR) if g.id == FIXTURE_ID)
    prompt = load_prompt("extractor")
    messages = build_messages(
        prompt_text=prompt.text,
        transcript=golden.transcript,
        date_iso=golden.expected.date.isoformat(),
        known_people=golden.known_people,
        known_places=golden.known_places,
        known_vocab=golden.known_vocab,
    )

    total_usd = 0.0
    field_sets: dict[str, set[str]] = {}
    for name, provider_cls, model in PROVIDERS:
        sheet, result = await provider_cls().structured(
            messages, BeatSheet, model=model, temperature=prompt.temperature
        )
        total_usd += result.usd
        field_sets[name] = set(sheet.model_dump().keys())
        print(f"{name} ({model}): {len(sheet.beats)} beats, ${result.usd:.5f}")

    names = list(field_sets)
    assert field_sets[names[0]] == field_sets[names[1]], (
        f"field sets differ: {names[0]}={field_sets[names[0]]} vs {names[1]}={field_sets[names[1]]}"
    )
    print(f"field sets identical: {sorted(field_sets[names[0]])}")
    print(f"total spend: ${total_usd:.5f} (cap ${SPEND_CAP_USD})")
    assert total_usd < SPEND_CAP_USD, f"spend ${total_usd:.5f} exceeded the ${SPEND_CAP_USD} cap"
    print("Gate 1: PASS")


if __name__ == "__main__":
    asyncio.run(main())
