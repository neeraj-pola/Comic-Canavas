"""Consistency critic — pipeline node 7.

Scores every candidate for a panel with the four critic signals (already
computed into `Candidate.scores` by the caller), combines them via the
reward head, and picks the winner. If every candidate for a panel fails the
identity hard threshold, the panel is regenerated (via the given callback) up
to `MAX_RETRIES` times, then falls back to the best-scoring candidate
available with an error appended to `DayState.errors` — a panel is never
silently dropped.

`IDENTITY_THRESHOLD = 0.55` sits between the measured mean score for a wrong
character (0.367) and the approved character (0.738).

Alignment is not used as an independent hard gate: SigLIP's absolute
confidence on this project's flat-ink comic style is very low across the
board, so a fixed absolute threshold would be meaningless — its *relative*
ordering is reliable, so it still contributes to ranking via the reward head.

Anti-hacking guards (`critic/guards.py`) apply on top of the base reward-head
score: a small bonus for `wide`/`medium` framings (offsetting identity's bias
toward close-up faces) and a penalty for detected in-panel text. Both no-op
when no script context is available, so this node's standalone tests are
unaffected.

`regenerate(panel_id)` is awaited to get a fresh batch of candidates for a
retry — this node only decides *when* to ask for one, not how.
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from app.critic.guards import guarded_reward
from app.critic.reward_head import score as reward_score
from app.preference.state import PreferenceState
from contracts import Candidate, DayState

IDENTITY_THRESHOLD = 0.55
MAX_RETRIES = 2

RegenerateFn = Callable[[int], Awaitable[list[Candidate]]]
Framing = Literal["wide", "medium", "close"]


def _stable_seed(job_id: str, panel_id: int) -> int:
    """Not `hash()` — that is randomized per process, which would make a
    resumed job re-pick differently."""
    return int.from_bytes(hashlib.sha256(f"{job_id}:{panel_id}".encode()).digest()[:4], "big")


@dataclass
class ScoredCandidate:
    candidate: Candidate
    reward: float


def passes_identity_gate(candidate: Candidate) -> bool:
    return candidate.scores.get("identity", 0.0) >= IDENTITY_THRESHOLD


def _reward(candidate: Candidate, *, framing: Framing | None, base: float | None = None) -> float:
    """`base` overrides the hand-set reward head with a learned reward when
    supplied; the framing/text guards apply either way."""
    has_text = candidate.scores.get("text_detected", 0.0) >= 0.5
    return guarded_reward(
        reward_score(candidate) if base is None else base, framing=framing, has_text=has_text
    )


async def choose_best_for_panel(
    panel_id: int,
    candidates: list[Candidate],
    *,
    regenerate: RegenerateFn,
    max_retries: int = MAX_RETRIES,
    framing: Framing | None = None,
    preference: PreferenceState | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[list[Candidate], list[str]]:
    """Returns `(final_candidates, errors)` — `final_candidates` is whichever
    batch was actually chosen from: the original `candidates` if no retry was
    needed, or the last retry's fresh batch otherwise (a retry discards the
    failed batch entirely). The winner has `chosen` set to `True` in place.
    The caller needs the whole final batch, not just the winner, to
    reconstruct `DayState.candidates` correctly."""
    errors: list[str] = []
    attempt = 0
    current = candidates

    while True:
        # Hand-set reward by default. Once the person's learned model is
        # active it ranks instead: `scores["reward"]` is the model's MEAN
        # reward (what the A/B runner-up is picked by), while the pick itself
        # is a Thompson sample from the posterior, so it explores exactly as
        # much as the model is unsure. `explored` records when the sample
        # disagreed with the mean's favourite.
        mean_base: list[float] | None = None
        sample_base: list[float] | None = None
        if preference is not None and preference.active:
            mean_base = preference.rewards(current, preference.mu)
            if mean_base is not None:
                generator = rng or np.random.default_rng(panel_id)
                sample_base = preference.rewards(current, preference.sample(generator))
        rewards_mean = [
            _reward(c, framing=framing, base=None if mean_base is None else mean_base[i])
            for i, c in enumerate(current)
        ]
        rewards_pick = (
            rewards_mean
            if sample_base is None
            else [_reward(c, framing=framing, base=sample_base[i]) for i, c in enumerate(current)]
        )
        scored = [ScoredCandidate(c, r) for c, r in zip(current, rewards_mean, strict=True)]
        any_pass = any(passes_identity_gate(s.candidate) for s in scored)

        if any_pass or attempt >= max_retries:
            # Persist the reward onto each candidate's `scores` (not just used
            # for local ranking) — the frontend's runner-up display reads it.
            for s in scored:
                s.candidate.scores["reward"] = s.reward
            pick_index = max(range(len(scored)), key=lambda i: rewards_pick[i])
            best = scored[pick_index]
            mean_favourite = max(range(len(scored)), key=lambda i: rewards_mean[i])
            best.candidate.scores["explored"] = 1.0 if pick_index != mean_favourite else 0.0
            if not any_pass:
                errors.append(
                    f"panel {panel_id}: no candidate passed the identity gate "
                    f"(threshold {IDENTITY_THRESHOLD}) after {attempt} retries — "
                    f"falling back to the best available candidate (reward "
                    f"{best.reward:.3f})"
                )
            best.candidate.chosen = True
            return current, errors

        attempt += 1
        current = await regenerate(panel_id)


async def choose_best_candidates(
    state: DayState,
    *,
    regenerate: RegenerateFn,
    preference: PreferenceState | None = None,
) -> DayState:
    """Groups `state.candidates` by `panel_id`, runs `choose_best_for_panel`
    for each group, and returns an updated `DayState` with `chosen` flags
    set, retried panels' candidates replaced by their final batch, and any
    fallback errors appended."""
    by_panel: dict[int, list[Candidate]] = {}
    for candidate in state.candidates:
        by_panel.setdefault(candidate.panel_id, []).append(candidate)

    framing_by_panel: dict[int, Framing] = {}
    if state.script is not None:
        framing_by_panel = {panel.id: panel.framing for panel in state.script.panels}

    final_candidates: list[Candidate] = []
    all_errors: list[str] = list(state.errors)
    for panel_id, panel_candidates in by_panel.items():
        panel_final, errors = await choose_best_for_panel(
            panel_id,
            panel_candidates,
            regenerate=regenerate,
            framing=framing_by_panel.get(panel_id),
            preference=preference,
            # Deterministic per job+panel so a resumed job re-picks the same.
            rng=np.random.default_rng(_stable_seed(state.job_id, panel_id)),
        )
        final_candidates.extend(panel_final)
        all_errors.extend(errors)

    return state.model_copy(update={"candidates": final_candidates, "errors": all_errors})
