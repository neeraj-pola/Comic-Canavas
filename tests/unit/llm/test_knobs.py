"""The knob model: reading a candidate's levels, the sweet-spot fit,
and the rules that move a person's default one level at a time."""

from __future__ import annotations

import numpy as np

from app.preference import knobs, model
from app.preference.axes import AXES, variant_prompt

K = knobs.K


def test_levels_come_from_the_recorded_knob_levels_of_a_fully_phrased_candidate() -> None:
    scores = {"level_warmth": 1.0, "level_closeness": -2.0, "level_expression": 0.0, "phrased": 1.0}
    assert list(knobs.levels_of(scores)) == [1.0, -2.0, 0.0]  # type: ignore[arg-type]


def test_candidates_where_the_default_was_a_bare_prompt_are_not_learned_from() -> None:
    """A plain "as usual" against phrased variants makes "any phrase beats
    none" look like a taste for warmth. Unphrased candidates are ignored,
    not guessed."""
    assert knobs.levels_of({"axis_id": 0.0, "axis_level": 1.0, "expression": 0.0}) is None
    unphrased = {"level_warmth": 1.0, "level_closeness": 0.0, "level_expression": 0.0}
    assert knobs.levels_of(unphrased) is None  # recorded levels, but not every option phrased
    assert knobs.levels_of({"identity": 0.5}) is None


def test_generated_candidates_record_every_knob_level_and_their_step_from_the_default() -> None:
    from app.preference.axes import plan_for_panel
    from contracts import ImagePrompt

    prompt = ImagePrompt(
        panel_id=2,
        generator="mock",
        character_clause="a man",
        environment_clause="a gym",
        positive="a man in a gym",
        negative="blurry",
        seed=100,
        guidance=None,
        steps=None,
    )
    plan = plan_for_panel(2, lean={"warmth": 1, "closeness": 0, "expression": -1})  # closeness
    presets = [variant_prompt(prompt, plan, k)[1] for k in range(3)]
    assert [p["offset"] for p in presets] == [-1.0, 0.0, 1.0]
    assert [p["level_closeness"] for p in presets] == [-1.0, 0.0, 1.0]
    assert all(p["level_warmth"] == 1.0 and p["level_expression"] == -1.0 for p in presets)
    assert [list(knobs.levels_of(p)) for p in presets][1] == [1.0, 0.0, -1.0]  # type: ignore[arg-type]


def _sweet_spot_utility(level: float) -> float:
    return -0.9 * (level - 1) ** 2


def _taps_from_sweet_spot(rng: np.random.Generator, n: int = 120) -> np.ndarray:
    """Pairs on the warmth knob from a person who likes level +1 best and dislikes going past it."""
    rows = []
    for _ in range(n):
        a, b = rng.choice(np.arange(-2, 3), size=2, replace=False)
        first_wins = rng.random() < 1 / (
            1 + np.exp(-(_sweet_spot_utility(a) - _sweet_spot_utility(b)))
        )
        win, lose = (a, b) if first_wins else (b, a)
        rows.append(knobs.phi(np.array([win, 0, 0])) - knobs.phi(np.array([lose, 0, 0])))
    return np.array(rows)


def test_the_squared_term_finds_a_sweet_spot_that_a_straight_line_runs_past() -> None:
    diffs = _taps_from_sweet_spot(np.random.default_rng(0))
    mu, cov = model.fit(diffs)
    assert knobs.best_level(mu, cov, 0) == 1  # the sweet spot

    linear_mu, _ = model.fit(diffs[:, :K])  # levels only, no squared terms
    linear_best = max(knobs.LEVELS, key=lambda level: linear_mu[0] * level)
    assert linear_best in (-2, 2)  # a straight line can only say "more" or "less"


def test_the_curve_is_relative_to_the_plain_prompt_and_the_band_narrows_with_evidence() -> None:
    mu, cov = np.zeros(2 * K), np.eye(2 * K)
    prior = {p["level"]: p for p in knobs.curve(mu, cov, 0)}
    assert prior[0.0]["mean"] == 0.0 and prior[0.0]["sd"] == 0.0
    fitted_mu, fitted_cov = model.fit(_taps_from_sweet_spot(np.random.default_rng(1)))
    fitted = {p["level"]: p for p in knobs.curve(fitted_mu, fitted_cov, 0)}
    assert fitted[1.0]["sd"] < prior[1.0]["sd"]
    assert fitted[1.0]["mean"] > 0  # +1 beats the plain prompt


def _sure_of(level_weight: float, square_weight: float, sd: float) -> tuple[np.ndarray, np.ndarray]:
    mu = np.zeros(2 * K)
    mu[0], mu[K] = level_weight, square_weight
    return mu, np.eye(2 * K) * sd**2


def test_no_default_moves_before_enough_taps() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.05)
    assert list(knobs.step_lean(np.zeros(K), mu, cov, knobs.MIN_TAPS_LEAN - 1)) == [0, 0, 0]
    assert list(knobs.step_lean(np.zeros(K), mu, cov, knobs.MIN_TAPS_LEAN)) == [1, 0, 0]


def test_a_default_moves_one_level_at_a_time_and_never_past_the_limit() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.05)
    lean = np.zeros(K, dtype=int)
    seen = []
    for _ in range(5):
        lean = knobs.step_lean(lean, mu, cov, 100)
        seen.append(int(lean[0]))
    assert seen == [1, 2, 2, 2, 2]


def test_an_unsure_model_does_not_move_the_default() -> None:
    mu, cov = _sure_of(0.5, 0.0, 1.0)  # a lean-worthy mean, but the band is wide
    assert list(knobs.step_lean(np.zeros(K), mu, cov, 100)) == [0, 0, 0]


def test_a_default_that_loses_its_support_steps_back_toward_the_plain_prompt() -> None:
    lean = np.array([2, 0, 0])
    mu, cov = _sure_of(0.0, 0.0, 1.0)  # nothing supports level 2 over level 1 any more
    assert list(knobs.step_lean(lean, mu, cov, 100)) == [1, 0, 0]
    assert list(knobs.step_lean(np.array([1, 0, 0]), mu, cov, 100)) == [0, 0, 0]


def test_a_default_that_is_still_well_supported_stays() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.05)
    assert list(knobs.step_lean(np.array([1, 0, 0]), mu, cov, 100)) == [2, 0, 0]  # keeps climbing
    mu2, cov2 = _sure_of(3.0, -1.5, 0.05)  # U = 3l - 1.5l^2: peaks at level 1, beating 0 and 2
    assert list(knobs.step_lean(np.array([1, 0, 0]), mu2, cov2, 100)) == [1, 0, 0]


def test_uncertainty_names_the_least_known_knob_and_falls_as_evidence_comes_in() -> None:
    cov = np.eye(2 * K)
    cov[0, 0] = cov[K, K] = 0.01  # warmth is well known
    unc = knobs.uncertainty(cov, np.zeros(K, dtype=int))
    assert set(unc) == set(AXES) and unc["warmth"] < unc["closeness"]


def test_forgetting_weights_halve_every_half_life() -> None:
    weights = model.decay_weights(np.array([0, 80, 160]), 80.0)
    assert list(np.round(weights, 3)) == [1.0, 0.5, 0.25]


def test_a_weighted_fit_follows_recent_taps_over_old_contradicting_ones() -> None:
    old = np.tile(-np.eye(1), (60, 1))  # 60 old taps against the feature
    new = np.tile(np.eye(1), (40, 1))  # 40 recent taps for it
    diffs = np.vstack([old, new])
    unweighted, _ = model.fit(diffs)
    ages = np.concatenate([np.arange(100, 40, -1), np.arange(40, 0, -1)])
    weighted, _ = model.fit(diffs, weights=model.decay_weights(ages, 20.0))
    assert unweighted[0] < 0 < weighted[0]


def test_a_default_never_moves_to_a_level_nobody_was_ever_shown() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.05)  # confident that more is better, forever
    seen = np.zeros((K, len(knobs.LEVELS)), dtype=bool)
    seen[0, 2 + np.array([-1, 0, 1])] = True  # only levels -1, 0, +1 were compared
    lean = np.zeros(K, dtype=int)
    for _ in range(4):
        lean = knobs.step_lean(lean, mu, cov, 100, seen)
    assert list(lean) == [1, 0, 0]  # stops at the edge of what it has seen, not at 2


def test_people_with_no_preference_rarely_get_any_default_at_all() -> None:
    """The false-alarm rate, with the model replayed after every tap as the app does: taps that
    are coin flips should rarely move a default (simulation: about 1 in 15)."""
    moved = 0
    people = 40
    for seed in range(people):
        rng = np.random.default_rng(seed)
        lean = np.zeros(K, dtype=int)
        rows: list[np.ndarray] = []
        ages: list[int] = []
        mu = np.zeros(2 * K)
        for t in range(1, 61):
            axis = (t - 1) % K
            levels = np.tile(lean.astype(float), (3, 1))
            for j, offset in enumerate((-1, 0, 1)):
                levels[j, axis] = np.clip(lean[axis] + offset, -2, 2)
            win = int(rng.integers(3))
            for other in range(3):
                if other != win:
                    rows.append(knobs.phi(levels[win]) - knobs.phi(levels[other]))
                    ages.append(t)
            weights = model.decay_weights(t - np.array(ages), knobs.HALF_LIFE_TAPS)
            mu, cov = model.fit(np.array(rows), w0=mu, weights=weights)
            lean = knobs.step_lean(lean, mu, cov, t)
        moved += int(np.any(lean != 0))
    assert moved / people <= 0.2


def test_the_best_guess_is_never_a_level_nobody_was_shown() -> None:
    """A convex fit ("the middle is worst, both sides are fine") is highest at the ends; without
    the guard it guessed 'much warmer' from picks that only ever reached +1."""
    mu = np.zeros(2 * K)
    mu[0], mu[K] = 0.5, 1.0  # U(l) = 0.5 l + l^2: level 2 looks best by extrapolation
    seen = np.zeros((K, len(knobs.LEVELS)), dtype=bool)
    seen[0, 2 + np.array([-1, 0, 1])] = True
    assert knobs.best_level(mu, np.eye(2 * K), 0) == 2
    assert knobs.best_level(mu, np.eye(2 * K), 0, seen) == 1
    flagged = {p["level"]: p["seen"] for p in knobs.curve(mu, np.eye(2 * K), 0, seen)}
    assert flagged == {-2.0: 0.0, -1.0: 1.0, 0.0: 1.0, 1.0: 1.0, 2.0: 0.0}


def test_the_best_guess_starts_from_the_confident_default_and_needs_ten_taps() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.3)
    lean = np.zeros(K, dtype=int)
    assert list(knobs.guess_levels(lean, mu, cov, knobs.MIN_TAPS_GUESS - 1)) == [0, 0, 0]
    assert list(knobs.guess_levels(lean, mu, cov, knobs.MIN_TAPS_GUESS)) == [1, 0, 0]


def test_the_best_guess_moves_one_level_and_only_when_fairly_sure() -> None:
    lean = np.zeros(K, dtype=int)
    sure = _sure_of(3.0, 0.0, 0.05)
    assert list(knobs.guess_levels(lean, *sure, 100)) == [1, 0, 0]  # one step, not two
    unsure = _sure_of(0.3, 0.0, 1.0)
    assert list(knobs.guess_levels(lean, *unsure, 100)) == [0, 0, 0]


def test_the_best_guess_never_steps_to_a_level_nobody_was_shown() -> None:
    mu, cov = _sure_of(3.0, 0.0, 0.05)
    seen = np.zeros((K, len(knobs.LEVELS)), dtype=bool)
    seen[0, 2] = True  # only the plain level was ever compared
    assert list(knobs.guess_levels(np.zeros(K, dtype=int), mu, cov, 100, seen)) == [0, 0, 0]


def test_the_best_guess_is_softer_than_the_confident_default() -> None:
    """A guess only shapes the centre option, so it needs 90% not 99% — a middling model gives a
    guess but no default."""
    mu, cov = _sure_of(1.0, 0.0, 0.5)  # ~92% sure: above the guess bar, below the default bar
    lean = knobs.step_lean(np.zeros(K, dtype=int), mu, cov, 100)
    guess = knobs.guess_levels(lean, mu, cov, 100)
    assert list(lean) == [0, 0, 0] and list(guess) == [1, 0, 0]
