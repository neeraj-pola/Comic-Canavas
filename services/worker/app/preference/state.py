"""The learned model as the pipeline consumes it (loaded once per job)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import psycopg

from app.preference import model
from app.preference.bandit import Posterior
from app.preference.embedding import Projection, project
from app.preference.features import FEATURES, feature_vector
from contracts import Candidate


@dataclass
class PreferenceState:
    mu: np.ndarray
    cov: np.ndarray
    scales: np.ndarray
    n_taps: int
    active: bool
    lean: dict[str, int] = field(default_factory=dict)
    z: dict[str, float] = field(default_factory=dict)
    # How unsure the knob model is about each style knob; the least-sure one is what a
    # panel's three options vary (see `axes.plan_for_panel`).
    knob_uncertainty: dict[str, float] = field(default_factory=dict)
    # The person's written-down taste (`notes.py`); empty until there are enough taps.
    notes: list[str] = field(default_factory=list)
    # The levels each panel's three options are drawn around: the confident default plus the
    # model's softer best guess (`knobs.guess_levels`). Empty when personalisation is off.
    guess: dict[str, int] = field(default_factory=dict)
    # The tempered knob posterior the option bandit draws from (`bandit.py`), and how many picks
    # it has learned from. `None` until the knob model has any.
    bandit: Posterior | None = None
    knob_taps: int = 0
    # The embedding projection this model was trained with (`embedding.py`).
    projection: Projection | None = None

    def vector(self, candidate: Candidate) -> np.ndarray | None:
        """The model's full feature vector for one candidate (its scores, then its embedding
        components); `None` if the measured features are missing."""
        base = feature_vector(candidate.scores)
        if base is None:
            return None
        return np.concatenate([base, project(self.projection, candidate.embedding)])

    def rewards(self, candidates: list[Candidate], w: np.ndarray) -> list[float] | None:
        """Learned reward in (0, 1) per candidate, centered within the panel
        (only the ranking matters, and the raw linear score can be large).
        `None` if any candidate lacks the measured features — the caller
        falls back to the hand-set reward for that panel."""
        vectors = [self.vector(c) for c in candidates]
        if any(v is None for v in vectors):
            return None
        raw = np.array([float(w @ (v / self.scales)) for v in vectors if v is not None])
        return [float(model.sigmoid(s - raw.mean())) for s in raw]

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        return model.sample_weights(self.mu, self.cov, rng)


def load_preference_state(conn: psycopg.Connection[Any], user_id: str) -> PreferenceState | None:
    """`None` when the user has no learned model yet. Returned even when
    `active` is false, since `z`/`n_taps` still steer which axis to explore."""
    row = conn.execute(
        "SELECT features, mu, cov, scales, n_taps, active, lean, z, knobs, notes, embed "
        "FROM preference_models WHERE user_id = %s",
        (user_id,),
    ).fetchone()
    if row is None:
        return None
    r = (
        row
        if isinstance(row, dict)
        else dict(
            zip(
                (
                    "features",
                    "mu",
                    "cov",
                    "scales",
                    "n_taps",
                    "active",
                    "lean",
                    "z",
                    "knobs",
                    "notes",
                ),
                row,
                strict=True,
            )
        )
    )
    # The person can switch personalisation off (Settings): then nothing they've taught it
    # steers a prompt or a script, though it keeps learning from taps.
    switch = conn.execute(
        "SELECT personalise FROM settings WHERE user_id = %s", (user_id,)
    ).fetchone()
    personalise = (
        True
        if switch is None
        else bool(switch["personalise"] if isinstance(switch, dict) else switch[0])
    )
    knobs = r["knobs"] or {}
    notes = r["notes"] or {}
    if list(r["features"]) != list(FEATURES):
        return None  # feature set changed since it was trained; ignore rather than misapply
    return PreferenceState(
        mu=np.array(r["mu"], dtype=np.float64),
        cov=np.array(r["cov"], dtype=np.float64),
        scales=np.array(r["scales"], dtype=np.float64),
        n_taps=int(r["n_taps"]),
        active=bool(r["active"]),
        lean={k: int(v) for k, v in (r["lean"] or {}).items()} if personalise else {},
        z={k: float(v) for k, v in (r["z"] or {}).items()},
        knob_uncertainty={k: float(v) for k, v in (knobs.get("uncertainty") or {}).items()},
        notes=[str(n) for n in (notes.get("items") or [])] if personalise else [],
        projection=Projection.from_json(r["embed"] or {}),
        guess={k: int(v) for k, v in (knobs.get("guess") or {}).items()} if personalise else {},
        bandit=Posterior.from_json(knobs.get("ts")) if personalise else None,
        knob_taps=int(knobs.get("n_taps") or 0),
    )
