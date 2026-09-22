"""Learning on the knobs the prompt actually turns.

The feature model (`model.py` over `features.py`) learns from measured pixel
statistics, which are noisy stand-ins for what we control. This module learns
directly on the three style knobs the prompt writer can move — warmth, framing,
expression — as integer levels -2..2, and decides the person's *default* level
for each ("the lean" baked into every new panel's prompt).

Three design choices, each settled by simulation (too few real taps to settle
them from data alone):

* **A squared term per knob.** A linear model can only say "more is better", so for
  someone with a sweet spot ("warm, but not too warm") it pushes the default to an
  extreme: their default got *worse* than doing nothing (regret 1.20 -> 2.64). With
  a level and a level-squared weight per knob it finds the middle (regret 0.17).
* **Stepwise, very confident moves.** A knob moves one level at a time, and only once the
  posterior is >= `ENTER_P` (99%) sure the new level beats the current one, and never
  before `MIN_TAPS_LEAN` taps. The model is re-checked after every tap, which invites
  false alarms: at 95% about a quarter of simulated people with NO preference at all were
  given a default; at 99% it is about 1 in 15, at the price of a slower start (a person
  with a strong sweet spot has a default on ~35% of runs by 60 taps, ~75% by 120; regret
  1.2 -> 0.9 -> 0.6). A wrong "your default is warmer" is worse than a late one.
* **A lean must keep earning its place.** Entering needs `ENTER_P`; staying needs only
  `KEEP_P`. Together with the forgetting factor (`model.decay_weights`) a lean that is no
  longer supported drifts back toward the plain prompt instead of sticking.

Pure numpy — no image or database code — so the simulation can import it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from app.preference.axes import AXES, MAX_LEVEL

K = len(AXES)
LEVELS = tuple(range(-MAX_LEVEL, MAX_LEVEL + 1))

# A comparison this many taps old counts half as much. Simulated: with 60-100 the default
# recovers from a taste change (regret 3.4 -> ~1.0 by 60 taps later, vs 1.8 without
# forgetting) at a small cost when taste is stable.
HALF_LIFE_TAPS = 80.0
ENTER_P = 0.99  # how sure before a knob's default moves one level away from 0
KEEP_P = 0.75  # how sure before a non-zero default is kept rather than stepped back
MIN_TAPS_LEAN = 20  # no default moves before this many taps
# The best guess (`guess_levels`): a softer bar, because the guess is only ever the CENTRE
# option of three — the person still sees the plain alternatives around it.
GUESS_P = 0.90
MIN_TAPS_GUESS = 10


def levels_of(scores: Mapping[str, float]) -> np.ndarray | None:
    """The three intended levels a candidate was generated with, or `None`.

    Only candidates generated with every option phrased (`phrased`) count. Earlier ones put
    a plain prompt ("as usual") against phrased variants, so "any phrase beats none" was
    indistinguishable from a real taste; they are ignored here rather than guessed at."""
    if "phrased" not in scores or not all(f"level_{axis}" in scores for axis in AXES):
        return None
    return np.array([float(scores[f"level_{axis}"]) for axis in AXES])


def phi(levels: np.ndarray) -> np.ndarray:
    """Level and level-squared per knob: `[l_1..l_K, l_1^2..l_K^2]`."""
    levels = np.asarray(levels, dtype=float)
    return np.concatenate([levels, levels * levels])


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _clamp(level: int) -> int:
    return max(-MAX_LEVEL, min(MAX_LEVEL, level))


def _axis_diff(axis: int, new: int, old: int) -> np.ndarray:
    """Feature difference `phi(new) - phi(old)` when only knob `axis` changes."""
    diff = np.zeros(2 * K)
    diff[axis] = new - old
    diff[K + axis] = new * new - old * old
    return diff


def p_better(mu: np.ndarray, cov: np.ndarray, axis: int, new: int, old: int) -> float:
    """Posterior probability that level `new` of knob `axis` beats level `old`."""
    diff = _axis_diff(axis, new, old)
    mean = float(mu @ diff)
    var = float(diff @ cov @ diff)
    if var <= 1e-12:
        return 1.0 if mean > 0 else 0.0
    return normal_cdf(mean / math.sqrt(var))


def step_lean(
    lean: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    n_taps: int,
    seen: np.ndarray | None = None,
) -> np.ndarray:
    """Move each knob's default at most one level. Away from zero needs `ENTER_P`
    (and `MIN_TAPS_LEAN` taps); a non-zero default steps back toward zero unless the
    posterior is still `KEEP_P` sure it beats the level nearer zero.

    `seen` (shape `(K, len(LEVELS))`, True where a candidate at that level has been
    compared) stops a default moving to a level nobody has ever been shown: the squared
    term can extrapolate a confident-looking "level 2 is better" from data that only
    reached level 1. In practice a level appears the tap after the default moves next to
    it (the three options are drawn around the default), so this only delays a step."""
    new = lean.astype(int).copy()
    for axis in range(K):
        current = int(new[axis])
        if n_taps >= MIN_TAPS_LEAN:
            best: tuple[int, float] | None = None
            for step in (-1, 1):
                candidate = _clamp(current + step)
                if candidate == current or abs(candidate) < abs(current):
                    continue  # only moves away from zero are "entering"
                if seen is not None and not seen[axis, candidate + MAX_LEVEL]:
                    continue
                p = p_better(mu, cov, axis, candidate, current)
                if p >= ENTER_P and (best is None or p > best[1]):
                    best = (candidate, p)
            if best is not None:
                new[axis] = best[0]
                continue
        if current != 0:
            nearer = current - int(np.sign(current))
            if p_better(mu, cov, axis, current, nearer) < KEEP_P:
                new[axis] = nearer
    return new


def guess_levels(
    lean: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    n_taps: int,
    seen: np.ndarray | None = None,
) -> np.ndarray:
    """The levels the app would show as its best guess right now: the confident default (`lean`),
    moved one level toward the model's best-seen level per knob when the posterior is
    >= `GUESS_P` sure that level beats the current one. Every panel's three options are drawn
    around this guess, so the centre option is always what the model thinks you would pick."""
    guess = lean.astype(int).copy()
    if n_taps < MIN_TAPS_GUESS:
        return guess
    for axis in range(K):
        best = best_level(mu, cov, axis, seen)
        if best == guess[axis]:
            continue
        target = _clamp(int(guess[axis]) + int(np.sign(best - guess[axis])))
        if seen is not None and not seen[axis, target + MAX_LEVEL]:
            continue
        if p_better(mu, cov, axis, target, int(guess[axis])) >= GUESS_P:
            guess[axis] = target
    return guess


def curve(
    mu: np.ndarray, cov: np.ndarray, axis: int, seen: np.ndarray | None = None
) -> list[dict[str, float]]:
    """Utility of each level of one knob relative to level 0: mean and standard deviation, and
    whether that level was ever compared (`seen` — 1.0/0.0; unseen levels are extrapolation)."""
    points = []
    for level in LEVELS:
        diff = _axis_diff(axis, level, 0)
        points.append(
            {
                "level": float(level),
                "mean": float(mu @ diff),
                "sd": math.sqrt(max(float(diff @ cov @ diff), 0.0)),
                "seen": 1.0 if seen is None or seen[axis, level + MAX_LEVEL] else 0.0,
            }
        )
    return points


def best_level(mu: np.ndarray, cov: np.ndarray, axis: int, seen: np.ndarray | None = None) -> int:
    """The level with the highest mean utility among the levels that were ever compared (the
    model's best guess, before any caution). A level nobody was shown is never a guess: a
    convex fit ("the middle is worst") otherwise extrapolates to "much warmer"."""
    points = [p for p in curve(mu, cov, axis, seen) if p["seen"] > 0]
    if not points:
        return 0
    return int(max(points, key=lambda p: p["mean"])["level"])


def uncertainty(cov: np.ndarray, lean: np.ndarray) -> dict[str, float]:
    """How unsure the model is about each knob around the current default: the standard
    deviation of `U(lean + 1) - U(lean - 1)`. The knob it knows least about is the one a
    panel's three options vary next."""
    out: dict[str, float] = {}
    for axis, name in enumerate(AXES):
        current = int(lean[axis])
        diff = _axis_diff(axis, _clamp(current + 1), _clamp(current - 1))
        out[name] = math.sqrt(max(float(diff @ cov @ diff), 0.0))
    return out
