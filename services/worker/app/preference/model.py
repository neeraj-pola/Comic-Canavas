"""Bayesian Bradley-Terry: which of two images does this person prefer?

Each tap is a pair (chosen, rejected). With `x = features(chosen) -
features(rejected)` (scaled), the model says P(chosen wins) = sigmoid(w . x).
The posterior over `w` is a Gaussian found by the Laplace approximation:
MAP via Newton's method with a N(0, prior_var * I) prior, covariance = the
inverse Hessian there. Ten features, so this is a 10x10 solve — microseconds
on CPU, which is why it can re-learn after every single tap for free.

The posterior gives us two things a plain point estimate can't: an honest
"how sure am I" per feature (the bars on the Learning page), and Thompson
sampling (draw a plausible `w`, act on it) so the app explores exactly as
much as it is unsure.
"""

from __future__ import annotations

import numpy as np

PRIOR_VAR = 1.0
NEWTON_ITERS = 30


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _objective(
    diffs: np.ndarray,
    w: np.ndarray,
    prior_prec: float | np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """Negative log posterior: sum weight * log(1 + exp(-x.w)) + 0.5 * prior_prec * |w|^2.
    Convex (a non-negative weight keeps every term convex)."""
    losses = np.logaddexp(0.0, -(diffs @ w))
    if weights is not None:
        losses = weights * losses
    return float(np.sum(losses) + 0.5 * float(np.sum(prior_prec * w * w)))


def decay_weights(ages: np.ndarray, half_life: float) -> np.ndarray:
    """Forgetting factor: a comparison `age` taps old counts for 0.5 ** (age / half_life)
    of a fresh one, so a taste that has changed stops being outvoted by old taps and a
    lean that is no longer supported fades instead of locking in forever."""
    return np.power(0.5, np.asarray(ages, dtype=float) / half_life)


def fit(
    diffs: np.ndarray,
    *,
    prior_var: float | np.ndarray = PRIOR_VAR,
    w0: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`diffs` is (n, d): scaled chosen-minus-rejected feature differences.
    Returns `(mu, cov)`. With no taps it returns the prior (0, prior_var*I).
    `weights` (one per row, default all 1) lets older comparisons count for less
    (see `decay_weights`). `prior_var` may be one number or one per weight.

    Damped Newton with a backtracking line search: a plain Newton step is not
    globally convergent for logistic regression — warm-starting from the
    previous fit after every tap, a full step can overshoot into a flat,
    saturated region and stay there. Since the objective is convex, insisting
    that each step actually lowers it (halving the step until it does) makes
    divergence impossible."""
    d = diffs.shape[1] if diffs.ndim == 2 else 0
    prior_prec = 1.0 / np.broadcast_to(np.asarray(prior_var, dtype=float), (d,))
    if diffs.shape[0] == 0:
        return np.zeros(d), np.diag(1.0 / prior_prec)
    w = np.zeros(d) if w0 is None else w0.copy()
    row_w = np.ones(diffs.shape[0]) if weights is None else weights
    f = _objective(diffs, w, prior_prec, row_w)
    for _ in range(NEWTON_ITERS):
        p = sigmoid(diffs @ w)
        grad = -(diffs.T @ (row_w * (1.0 - p))) + prior_prec * w
        curvature = row_w * p * (1.0 - p)
        hessian = (diffs.T * curvature) @ diffs + np.diag(prior_prec)
        step = np.linalg.solve(hessian, grad)
        slope = float(grad @ step)  # > 0: the Hessian is positive definite, so -step descends
        t = 1.0
        while True:
            candidate = w - t * step
            f_candidate = _objective(diffs, candidate, prior_prec, row_w)
            if f_candidate <= f - 1e-4 * t * slope:
                break
            t *= 0.5
            if t < 1e-12:
                candidate, f_candidate = w, f
                break
        moved = float(np.max(np.abs(candidate - w)))
        w, f = candidate, f_candidate
        if moved < 1e-9:
            break
    p = sigmoid(diffs @ w)
    hessian = (diffs.T * (row_w * p * (1.0 - p))) @ diffs + np.diag(prior_prec)
    return w, np.linalg.inv(hessian)


def predict_prob(mu: np.ndarray, cov: np.ndarray, diff: np.ndarray) -> float:
    """P(chosen wins) integrating over the posterior (probit approximation):
    an unsure model is pulled toward 0.5 instead of being overconfident."""
    mean = float(mu @ diff)
    var = float(diff @ cov @ diff)
    return float(sigmoid(mean / np.sqrt(1.0 + np.pi * var / 8.0)))


def sample_weights(mu: np.ndarray, cov: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Thompson sampling: one plausible weight vector from the posterior."""
    try:
        chol = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        chol = np.linalg.cholesky(cov + np.eye(len(mu)) * 1e-9)
    return mu + chol @ rng.standard_normal(len(mu))
