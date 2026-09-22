"""Feel-Good Thompson Sampling for dueling bandits: which three settings a panel is
drawn with — the best guess plus the best setting of two independent posterior draws."""

from __future__ import annotations

import numpy as np
import pytest

from app.nodes.generate import _bandit_variants
from app.preference import bandit, knobs
from app.preference.axes import AXES, plan_for_panel
from app.preference.features import FEATURES
from app.preference.state import PreferenceState
from contracts import ImagePrompt

K = knobs.K


def _posterior(sd: float, *, mean: np.ndarray | None = None) -> bandit.Posterior:
    mu = np.zeros(2 * K) if mean is None else mean
    return bandit.Posterior(mu=mu, cov=np.eye(2 * K) * sd**2, center_mu=mu.copy())


def test_the_arms_are_the_settings_within_one_level_of_the_guess() -> None:
    assert len(bandit.arm_levels(np.zeros(K))) == 27
    corner = bandit.arm_levels(np.array([2, 2, 2]))
    assert len(corner) == 8 and corner.min() == 1 and corner.max() == 2  # clamped to -2..2


def test_a_posterior_survives_json_including_what_the_feel_good_term_needs() -> None:
    post = bandit.Posterior(
        mu=np.arange(6) / 10,
        cov=np.eye(6) * 0.5,
        center_mu=np.arange(6) / 5,
        other_arm={1: np.ones((2, 6)), 2: np.zeros((3, 6))},
    )
    back = bandit.Posterior.from_json(post.to_json())
    assert back is not None and back.center_mu is not None
    assert np.allclose(back.mu, post.mu) and np.allclose(back.center_mu, post.center_mu)  # type: ignore[arg-type]
    assert back.other_arm[1].shape == (2, 6) and back.other_arm[2].shape == (3, 6)
    assert bandit.Posterior.from_json({}) is None and bandit.Posterior.from_json(None) is None


def test_the_options_are_three_distinct_settings_led_by_the_best_guess() -> None:
    mean = np.zeros(2 * K)
    mean[0] = 1.0  # warmer is better: the centre must be the warmest reachable setting
    post = _posterior(0.7, mean=mean)
    chosen = bandit.choose_options(post, np.zeros(K, dtype=int), np.random.default_rng(0))
    assert len(chosen) == 3
    assert len({tuple(levels) for levels, _ in chosen}) == 3
    centre, role = chosen[0]
    assert role == 0 and centre[0] == 1  # one level warmer than the guess, the edge of the arms
    assert all(np.abs(levels).max() <= 1 for levels, _ in chosen)  # within one level of a 0 guess


def test_the_two_draws_differ_from_each_other_across_rounds_when_unsure() -> None:
    post = _posterior(1.0)
    seen = set()
    for seed in range(30):
        chosen = bandit.choose_options(post, np.zeros(K, dtype=int), np.random.default_rng(seed))
        seen |= {tuple(levels) for levels, role in chosen if role in (1, 2)}
    assert len(seen) > 6  # it explores widely while it does not know


def test_a_sure_model_stops_exploring_and_the_fallback_keeps_three_distinct_options() -> None:
    mean = np.zeros(2 * K)
    mean[0], mean[K + 0] = 4.0, -2.0  # warmth: a firm sweet spot at level 1
    post = _posterior(0.02, mean=mean)
    fallback = [np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
    chosen = bandit.choose_options(
        post, np.zeros(K, dtype=int), np.random.default_rng(1), fallback=fallback
    )
    assert len({tuple(levels) for levels, _ in chosen}) == 3
    exploring = [levels for levels, role in chosen if role in (1, 2)]
    assert all(levels[0] == 1 for levels in exploring)  # every draw agrees on the sweet spot


def test_options_are_reproducible_for_the_same_generator_state() -> None:
    post = _posterior(1.0)
    first = bandit.choose_options(post, np.zeros(K, dtype=int), np.random.default_rng(5))
    second = bandit.choose_options(post, np.zeros(K, dtype=int), np.random.default_rng(5))
    assert [(list(a), r) for a, r in first] == [(list(a), r) for a, r in second]


def test_the_feel_good_term_favours_draws_that_see_a_strong_arm() -> None:
    """The exploration bonus: parameters that believe some arm is very good are more likely."""
    post = _posterior(1.0)
    post.other_arm = {1: np.zeros((5, 2 * K))}  # the other sampler's arm had zero utility
    phi = bandit.arm_features(bandit.arm_levels(np.zeros(K, dtype=int)))

    def mean_best(mu_fg: float) -> float:
        rng = np.random.default_rng(0)
        draws = [bandit.sample_theta(post, 1, phi, rng, mu_fg=mu_fg) for _ in range(300)]
        return float(np.mean([(phi @ d).max() for d in draws]))

    assert mean_best(1.0) > mean_best(0.0)


def _prompt() -> ImagePrompt:
    return ImagePrompt(
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


def _state(*, knob_taps: int, sd: float = 1.0, with_bandit: bool = True) -> PreferenceState:
    return PreferenceState(
        mu=np.zeros(len(FEATURES)),
        cov=np.eye(len(FEATURES)),
        scales=np.ones(len(FEATURES)),
        n_taps=knob_taps,
        active=False,
        bandit=_posterior(sd) if with_bandit else None,
        knob_taps=knob_taps,
    )


def test_the_bandit_only_takes_over_after_enough_picks_and_never_without_personalisation() -> None:
    plan = plan_for_panel(2)
    assert _bandit_variants(_prompt(), plan, None, "job") is None
    assert (
        _bandit_variants(_prompt(), plan, _state(knob_taps=bandit.MIN_TAPS_BANDIT - 1), "j") is None
    )
    assert _bandit_variants(_prompt(), plan, _state(knob_taps=20, with_bandit=False), "j") is None


def test_bandit_options_carry_their_levels_role_and_one_seed_and_are_stable_per_job() -> None:
    plan = plan_for_panel(2)
    state = _state(knob_taps=20)
    made = _bandit_variants(_prompt(), plan, state, "job-1")
    assert made is not None and len(made) == 3
    presets = [preset for _, preset in made]
    assert sorted(p["role"] for p in presets) == [0.0, 1.0, 2.0]
    assert [p["offset"] for p in presets if p["role"] == 0.0] == [0.0]
    assert all(p["phrased"] == 1.0 and all(f"level_{a}" in p for a in AXES) for p in presets)
    assert len({v.positive for v, _ in made}) == 3  # three different prompts
    assert {v.seed for v, _ in made} == {100}  # one seed: they differ only in the phrase
    again = _bandit_variants(_prompt(), plan, state, "job-1")
    assert again is not None and [v.positive for v, _ in again] == [v.positive for v, _ in made]


def test_the_environment_switch_returns_to_the_structured_design(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IMAGE_OPTIONS", "structured")
    assert _bandit_variants(_prompt(), plan_for_panel(2), _state(knob_taps=20), "job") is None
