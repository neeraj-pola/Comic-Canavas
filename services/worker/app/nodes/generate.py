"""Panel generator — pipeline node 6.

Resolves the configured `ImageProvider`, checks the cost guard before
generating anything, then fans out all panels' generation concurrently under
a semaphore and collects every panel's candidates onto `DayState.candidates`.
"""

from __future__ import annotations

import asyncio
import hashlib
import os

import numpy as np
from storage import Storage

from app.images import routing
from app.images.base import IdentityRef
from app.llm.cost import CostEvent, get_cost_sink
from app.preference import bandit
from app.preference.axes import (
    AXES,
    AxisPlan,
    levels_for_candidate,
    option_prompts,
    plan_for_panel,
    variant_prompt,
)
from app.preference.state import PreferenceState
from contracts import Candidate, DayState, ImagePrompt

DEFAULT_CANDIDATES_PER_PANEL = 3
DEFAULT_CONCURRENCY = 6
DEFAULT_MAX_USD_PER_DAY = 0.60


class NoPromptsError(ValueError):
    def __init__(self) -> None:
        super().__init__("generate_panels: DayState.prompts is empty")


class CostGuardExceededError(ValueError):
    """Raised instead of calling any provider when the projected image cost
    for this job already exceeds the cap."""

    def __init__(self, projected_usd: float, cap_usd: float) -> None:
        super().__init__(
            f"projected image cost ${projected_usd:.2f} exceeds "
            f"MAX_USD_PER_DAY (${cap_usd:.2f}) — job aborted before generating"
        )
        self.projected_usd = projected_usd
        self.cap_usd = cap_usd


def _max_usd_per_day() -> float:
    return float(os.environ.get("MAX_USD_PER_DAY", DEFAULT_MAX_USD_PER_DAY))


async def generate_one_panel(
    prompt: ImagePrompt,
    *,
    storage: Storage,
    identity: IdentityRef,
    n: int = DEFAULT_CANDIDATES_PER_PANEL,
) -> list[Candidate]:
    """Generates candidates for exactly one already-built `ImagePrompt` — the
    critic's retry path, used after `write_single_prompt` rebuilds that one
    panel's prompt. No cost-guard check here: the guard is a whole-day
    projection made once before the initial batch; a single-panel retry is a
    small marginal cost, not a fresh projection to gate."""
    provider = routing.resolve(storage)
    candidates = await provider.generate(prompt, n=n, identity=identity)
    get_cost_sink().record(
        CostEvent(
            provider=provider.name,
            model=prompt.generator,
            role="image",
            units=len(candidates),
            usd=provider.estimate_usd(n=n),
        )
    )
    return candidates


def _same_seed() -> bool:
    """The three options of a panel share one seed by default, so the person compares what the
    prompt changed rather than three different random draws. `IMAGE_SAME_SEED=0` restores a
    different seed per option."""
    return os.environ.get("IMAGE_SAME_SEED", "1") != "0"


def _bandit_variants(
    prompt: ImagePrompt, plan: AxisPlan, preference: PreferenceState | None, job_id: str
) -> list[tuple[ImagePrompt, dict[str, float]]] | None:
    """The panel's three options chosen by Feel-Good Thompson Sampling for dueling bandits
    (`preference/bandit.py`): the model's best guess plus the best setting of each of two
    independent posterior draws, all within one level of the guess. `None` — the structured
    design (one knob stepped either way) — until the knob model has `MIN_TAPS_BANDIT` picks,
    or when personalisation is off. Deterministic per job and panel, so a resumed job draws
    the same options."""
    if (
        preference is None
        or preference.bandit is None
        or preference.knob_taps < bandit.MIN_TAPS_BANDIT
        or os.environ.get("IMAGE_OPTIONS", "bandit") != "bandit"
    ):
        return None
    guess = np.array([(preference.guess or preference.lean).get(a, 0) for a in AXES])
    digest = hashlib.sha256(f"{job_id}:{prompt.panel_id}:options".encode()).hexdigest()
    rng = np.random.default_rng(int(digest[:8], 16))
    fallback = [
        np.array([levels_for_candidate(plan, k, 3)[a] for a in AXES], dtype=float) for k in (0, 2)
    ]
    chosen = bandit.choose_options(preference.bandit, guess, rng, fallback=fallback)
    if len(chosen) < 3:
        return None
    level_sets = [{a: int(levels[i]) for i, a in enumerate(AXES)} for levels, _ in chosen]
    return option_prompts(prompt, level_sets, [role for _, role in chosen], same_seed=_same_seed())


async def generate_panels(
    state: DayState,
    *,
    storage: Storage,
    identity: IdentityRef,
    n: int = DEFAULT_CANDIDATES_PER_PANEL,
    concurrency: int = DEFAULT_CONCURRENCY,
    preference: PreferenceState | None = None,
) -> DayState:
    if not state.prompts:
        raise NoPromptsError

    provider = routing.resolve(storage)

    cap_usd = _max_usd_per_day()
    projected_usd = provider.estimate_usd(n=n) * len(state.prompts)
    if projected_usd > cap_usd:
        raise CostGuardExceededError(projected_usd, cap_usd)

    semaphore = asyncio.Semaphore(concurrency)

    async def _generate_batch(prompt: ImagePrompt, count: int) -> list[Candidate]:
        async with semaphore:
            return await provider.generate(prompt, n=count, identity=identity)

    def _record_cost(prompt: ImagePrompt, candidates: list[Candidate]) -> None:
        # One cost event per panel, however many provider calls it took.
        get_cost_sink().record(
            CostEvent(
                provider=provider.name,
                model=prompt.generator,
                role="image",
                units=len(candidates),
                usd=provider.estimate_usd(n=n),
            )
        )

    async def _generate_one(prompt: ImagePrompt) -> list[Candidate]:
        if n != DEFAULT_CANDIDATES_PER_PANEL:
            candidates = await _generate_batch(prompt, n)
            _record_cost(prompt, candidates)
            return candidates
        # The panel's three candidates vary along ONE axis, chosen by what the
        # model is least sure about, so a tap teaches a direction. Seeds stay
        # `base + k`, the same as a plain n=3 call would produce.
        plan = plan_for_panel(
            prompt.panel_id,
            z=preference.z if preference else None,
            lean=(preference.guess or preference.lean) if preference else None,
            n_taps=preference.n_taps if preference else 0,
            uncertainty=preference.knob_uncertainty if preference else None,
        )
        variants = _bandit_variants(prompt, plan, preference, state.job_id)
        if variants is None:
            variants = [
                variant_prompt(prompt, plan, k, n, same_seed=_same_seed()) for k in range(n)
            ]
        batches = await asyncio.gather(*(_generate_batch(v, 1) for v, _ in variants))
        candidates = [
            candidate.model_copy(update={"scores": dict(preset)})
            for (_, preset), batch in zip(variants, batches, strict=True)
            for candidate in batch
        ]
        _record_cost(prompt, candidates)
        return candidates

    results = await asyncio.gather(*(_generate_one(prompt) for prompt in state.prompts))
    all_candidates = [candidate for panel_candidates in results for candidate in panel_candidates]

    return state.model_copy(
        update={
            "candidates": all_candidates,
            "cost_usd": state.cost_usd + projected_usd,
        }
    )
