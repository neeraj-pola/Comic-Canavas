"""Which three settings to draw for a panel: Feel-Good Thompson Sampling for dueling bandits.

The knob model (`knobs.py`) knows how much the person likes each level of warmth, framing and
expression. Deciding which settings to *show* is a contextual dueling bandit: two options are
compared, one preferred, and we want the person's picks to teach the most while the options stay
good. Li et al., "Feel-Good Thompson Sampling for Contextual Dueling Bandits" (ICML 2024,
arXiv 2404.06013), FGTS.CDB:

* the preference between two arms is Bradley-Terry over a linear utility, the same model
  we already learn;
* each round draw **two parameters independently** from the (tempered) posterior and take each
  one's best arm — no constraint that the second differs from the first;
* the posterior for parameter j is  p_0(theta) * exp(-sum_i L^j_i(theta)),  with
  L^j_i = eta * log(1 + exp(-y <theta, phi(a1_i) - phi(a2_i)>))
          - mu * max_{a'} <theta, phi(a') - phi(a_i^{3-j})>
  — the likelihood is tempered by eta (0.25 in the paper) and the "feel-good" term rewards
  parameters that believe some arm is good, the exploration bonus behind the regret bound
  O~(d sqrt(T)) (nearly minimax-optimal);
* the paper samples with SGLD.

**What this implementation is, and is not.** Real assumptions of the guarantee that we only
approximate — so the bound is motivation, not a promise: (1) the posterior is a Gaussian
(Laplace fit, tempered by eta through the row weights) reweighted by the feel-good term with
importance sampling, not exact or SGLD samples; (2) the arm set is the 27 settings within one
level of the current best guess (radius 1), not all 125 — so regret is measured against that
set, and it moves with the guess; (3) the feedback is a best-of-3 pick, learned as the pick beating
each other option (rank-breaking), not one pair per round; (4) old picks are forgotten
(`model.decay_weights`), which the theory does not cover; (5) the third option is the model's
own best guess, not part of the algorithm. What *is* checked is behaviour: unit tests and a
simulation of regret against the previous, non-bandit design.

Pure numpy — no image or database code.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from app.preference import knobs
from app.preference.axes import AXES, MAX_LEVEL

K = len(AXES)
ETA = 0.25  # likelihood tempering, the paper's value
RADIUS = 1  # arms are the settings within this many levels of the best guess, per knob
SAMPLES = 400  # Gaussian draws reweighted by the feel-good term
MIN_TAPS_BANDIT = 10  # before this the panel's options step one knob, as before (structured)
# Feel-good weight (the paper's mu): reward for parameters that see a strong arm. The paper's
# mu = 1 / (10 e^B sqrt(T)) is a worst-case constant that is ~0 at any realistic B; simulated
# over many people and picks, mu in 0.005-0.02 performed the same as no feel-good term at all
# (0.1 was worse on regret) — the gain comes from the tempered two-draw dueling design, not
# from this bonus.
FEEL_GOOD_MU = 0.005


def arm_levels(center: np.ndarray, radius: int = RADIUS) -> np.ndarray:
    """All settings within `radius` levels of `center` on every knob, clamped to -2..2. (M, K)."""
    ranges = [
        range(max(-MAX_LEVEL, int(c) - radius), min(MAX_LEVEL, int(c) + radius) + 1) for c in center
    ]
    return np.array(list(itertools.product(*ranges)), dtype=float)


def arm_features(levels: np.ndarray) -> np.ndarray:
    """(M, 2K) knob features for a set of arms."""
    return np.array([knobs.phi(row) for row in np.atleast_2d(levels)])


@dataclass
class Posterior:
    """The tempered knob posterior plus what the feel-good term needs to know about past rounds."""

    mu: np.ndarray
    cov: np.ndarray
    # The plain (untempered) posterior mean — the model's real best estimate. The tempered draw
    # spreads exploration; the best guess (option 0) must not be spread.
    center_mu: np.ndarray | None = None
    # For parameter j (1 or 2): the features of the arm the OTHER sampler's option had in each
    # past round it is known, i.e. phi(a_i^{3-j}).
    other_arm: dict[int, np.ndarray] = field(default_factory=dict)

    def to_json(self) -> dict[str, object]:
        return {
            "mu": [round(float(x), 6) for x in self.mu],
            "center_mu": [round(float(x), 6) for x in self.center_mu]
            if self.center_mu is not None
            else None,
            "cov": [[round(float(x), 6) for x in row] for row in self.cov],
            "other_arm": {
                str(j): [[round(float(x), 5) for x in row] for row in rows]
                for j, rows in self.other_arm.items()
            },
        }

    @staticmethod
    def from_json(data: dict[str, object] | None) -> Posterior | None:
        if not data or "mu" not in data or "cov" not in data:
            return None
        others = data.get("other_arm") or {}
        assert isinstance(others, dict)
        return Posterior(
            mu=np.array(data["mu"], dtype=np.float64),
            cov=np.array(data["cov"], dtype=np.float64),
            center_mu=np.array(data["center_mu"], dtype=np.float64)
            if data.get("center_mu")
            else None,
            other_arm={
                int(j): np.array(rows, dtype=np.float64).reshape(-1, 2 * K)
                for j, rows in others.items()
            },
        )


def sample_theta(
    posterior: Posterior,
    j: int,
    arm_phi: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_fg: float = FEEL_GOOD_MU,
    samples: int = SAMPLES,
) -> np.ndarray:
    """One parameter draw for sampler `j`: a tempered-Laplace Gaussian draw, importance-reweighted
    by exp(mu * sum_i max_{a'} <theta, phi(a') - phi(a_i^{3-j})>) when `mu_fg` > 0."""
    try:
        chol = np.linalg.cholesky(posterior.cov + np.eye(len(posterior.mu)) * 1e-9)
    except np.linalg.LinAlgError:
        chol = np.diag(np.sqrt(np.clip(np.diag(posterior.cov), 1e-9, None)))
    draws = posterior.mu + rng.standard_normal((samples, len(posterior.mu))) @ chol.T
    others = posterior.other_arm.get(j)
    if mu_fg <= 0.0 or others is None or len(others) == 0:
        return np.asarray(draws[0])
    best = (draws @ arm_phi.T).max(axis=1)  # (S,) best available utility under each draw
    other_utilities = draws @ others.T  # (S, R)
    log_w = mu_fg * (best[:, None] - other_utilities).sum(axis=1)
    log_w -= log_w.max()
    w = np.exp(log_w)
    return np.asarray(draws[rng.choice(samples, p=w / w.sum())])


def choose_options(
    posterior: Posterior,
    guess: np.ndarray,
    rng: np.random.Generator,
    *,
    mu_fg: float = FEEL_GOOD_MU,
    fallback: list[np.ndarray] | None = None,
) -> list[tuple[np.ndarray, int]]:
    """Three distinct settings with their role: (0) the best guess — the arm with the highest
    posterior-mean utility among the arms around the guess — then the best arm of each of two
    independent posterior draws (roles 1 and 2). A repeated setting is replaced from
    `fallback` (single-knob steps around the guess) so the three are always different."""
    arms = arm_levels(guess)
    phi = arm_features(arms)
    centre_mu = posterior.center_mu if posterior.center_mu is not None else posterior.mu
    center = arms[int(np.argmax(phi @ centre_mu))]
    picks = [(center, 0)]
    for j in (1, 2):
        theta = sample_theta(posterior, j, phi, rng, mu_fg=mu_fg)
        picks.append((arms[int(np.argmax(phi @ theta))], j))
    chosen: list[tuple[np.ndarray, int]] = []
    for levels, role in picks:
        if not any(np.array_equal(levels, c) for c, _ in chosen):
            chosen.append((levels, role))
    spare = list(fallback or [])
    for levels in spare:
        if len(chosen) >= 3:
            break
        if not any(np.array_equal(levels, c) for c, _ in chosen):
            chosen.append((levels, len(chosen)))
    # Still short (a one-knob-wide neighbourhood at the limits): any other arm nearby.
    for levels in arms:
        if len(chosen) >= 3:
            break
        if not any(np.array_equal(levels, c) for c, _ in chosen):
            chosen.append((levels, len(chosen)))
    return chosen[:3]
