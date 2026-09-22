"""Rebuild a person's preference model — and the snapshots the Learning page
charts — from every A/B tap they have made. Runs after each tap.

It is a full rebuild (not an incremental update) on purpose: with at most a
few hundred taps and ten features it costs milliseconds, and rebuilding from
the source rows means the snapshots can never drift from the taps.

**Held-out honesty (the gate).** For every tap the model is first asked "which
would you pick?" using only the taps BEFORE it (prequential evaluation), and
only then learns from it. The hand-set weights are asked the same question. The
learned model replaces the hand-set weights in the pipeline only once it has
enough taps and has beaten the hand-set weights on those held-out predictions.

**A tap is a pick of the favourite of a panel's three options** (best-of-3). It is learned
as the pick beating each of the other options, so one tap is two comparisons — twice the
information of a plain pair, for the same effort. Older taps (one pair) still count, as a
single comparison.

**Forgetting.** Every comparison is weighted by how many taps ago it was made
(`model.decay_weights`, half-life `knobs.HALF_LIFE_TAPS`), so a taste that changes is not
outvoted forever by old taps.

**Two models learn from the same taps.** The feature model (ten measured features) is what
ranks candidates once it beats the default ranking. The knob model (`knobs.py`, three
prompt knobs with a sweet spot each) decides the person's default level per knob — the lean
baked into every new prompt — and is judged on its own, not on beating the ranking: "which
direction to nudge" is a different question from "which of three images is best".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import psycopg
from psycopg.types.json import Json
from storage import Storage

from app.critic.reward_head import score as hand_reward
from app.preference import bandit, embedding, knobs, model
from app.preference.axes import AXES, MAX_LEVEL
from app.preference.features import (
    DEFAULT_SCALES,
    FEATURES,
    SCORE_FEATURES,
    feature_vector,
    pixel_features,
    prior_variances,
)
from contracts import Candidate
from ml.identity.faces import decode_jpeg_bgr

MIN_TAPS = 20  # never activate before this many taps
# The learned model must beat the default by at least this many NET calls (a tap it gets
# right that the default gets wrong counts +1, the reverse -1, ties half). "Strictly more
# accurate" was far too easy on ~20 taps: one lucky tap flipped it on (a 71% score on 21
# taps has a 95% margin of about +-20%). Three net calls is still a small bar, but it takes
# a real, repeated edge rather than a coin flip.
MIN_EDGE = 3.0
WARMUP = 5  # taps the model is allowed to see before its predictions count
MAX_TAPS = 500
N_THRESHOLDS = 2  # rating cut points: off|ok and ok|great
THRESHOLD_PRIOR_VAR = 4.0  # wide: where the cut points sit is entirely up to the ratings


@dataclass
class Tap:
    day_id: str
    panel_id: int
    tapped_at: Any
    chosen_url: str
    rejected_urls: list[str]
    chosen: np.ndarray
    rejected: list[np.ndarray]
    chosen_role: int | None
    rejected_roles: list[int | None]
    chosen_emb: list[float] | None
    rejected_emb: list[list[float] | None]
    chosen_levels: np.ndarray | None
    rejected_levels: list[np.ndarray | None]
    hand_correct: float
    axis: int | None
    default_picked: bool | None


def _candidate(
    conn: psycopg.Connection[Any], storage: Storage | None, day_id: str, url: str
) -> Candidate | None:
    row = conn.execute(
        "SELECT id, panel_id, url, seed, scores, face_box, embedding FROM candidates "
        "WHERE day_id = %s AND url = %s LIMIT 1",
        (day_id, url),
    ).fetchone()
    if row is None:
        return None
    scores = dict(row["scores"] or {})
    candidate = Candidate(
        id=row["id"],
        panel_id=row["panel_id"],
        url=row["url"],
        seed=row["seed"],
        scores=scores,
        face_box=tuple(row["face_box"]) if row["face_box"] else None,
        embedding=row["embedding"],
    )
    if candidate.embedding is None and storage is not None:
        # Scored before embeddings were kept: embed the stored image once and remember it.
        try:
            from ml.identity.style_score import embed_image

            vector = embed_image(decode_jpeg_bgr(storage.get_object_by_url(url)))
            candidate.embedding = [round(float(x), 5) for x in vector]
            conn.execute(
                "UPDATE candidates SET embedding = %s WHERE day_id = %s AND url = %s",
                (Json(candidate.embedding), day_id, url),
            )
        except Exception:
            candidate.embedding = None
    if (
        feature_vector(scores) is None
        and storage is not None
        and scores.get("identity") is not None
    ):
        # A candidate scored before the pixel features existed: measure them
        # now from the stored image rather than throwing the tap away.
        try:
            image = decode_jpeg_bgr(storage.get_object_by_url(url))
            candidate.scores.update(pixel_features(image, candidate.face_box))
        except Exception:
            return None
    return candidate


def _hit_among(scores: list[float]) -> float:
    """1.0 if index 0 is the strict maximum, 1/k if it ties with k-1 others for it, else 0."""
    top = max(scores)
    winners = [i for i, s in enumerate(scores) if s >= top - 1e-12]
    return 1.0 / len(winners) if 0 in winners else 0.0


def _role(scores: dict[str, float]) -> int | None:
    return int(scores["role"]) if "role" in scores else None


def _was_default(scores: dict[str, float]) -> bool | None:
    """Did this candidate carry the person's default (no step either side)? `None` when
    unknown — including candidates from before every option was phrased (`phrased`), whose
    "default" was a bare prompt the others were compared against."""
    if "offset" in scores and "phrased" in scores:
        return abs(scores["offset"]) < 1e-9
    return None


def _load_taps(conn: psycopg.Connection[Any], user_id: str, storage: Storage | None) -> list[Tap]:
    # One tap per (day, panel): the latest one. A best-of-3 tap wrote one row per option
    # passed on, all sharing a `tap_id`; an older single-pair tap is a group of one.
    rows = conn.execute(
        """
        WITH latest AS (
            SELECT DISTINCT ON (ip.day_id, ip.panel_id)
                   ip.day_id, ip.panel_id, ip.created_at,
                   COALESCE(ip.tap_id, ip.id::text) AS tid
            FROM image_pairs ip JOIN days d ON d.job_id = ip.day_id
            WHERE d.user_id = %s AND d.kind = 'daily' AND ip.source = 'ab'
            ORDER BY ip.day_id, ip.panel_id, ip.created_at DESC, ip.id DESC
        )
        SELECT l.day_id, l.panel_id, l.created_at, ip.chosen, ip.rejected
        FROM latest l JOIN image_pairs ip
          ON ip.day_id = l.day_id AND ip.panel_id = l.panel_id
         AND COALESCE(ip.tap_id, ip.id::text) = l.tid
        ORDER BY l.created_at, l.day_id, l.panel_id, ip.id
        """,
        (user_id,),
    ).fetchall()
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in rows:
        key = (row["day_id"], row["panel_id"])
        group = grouped.setdefault(
            key, {"created_at": row["created_at"], "chosen": row["chosen"], "rejected": []}
        )
        if row["rejected"] not in group["rejected"]:
            group["rejected"].append(row["rejected"])
    ordered = sorted(grouped.items(), key=lambda kv: kv[1]["created_at"])[-MAX_TAPS:]

    taps: list[Tap] = []
    for (day_id, panel_id), group in ordered:
        chosen = _candidate(conn, storage, day_id, group["chosen"])
        others = [_candidate(conn, storage, day_id, url) for url in group["rejected"]]
        if chosen is None or any(o is None for o in others):
            continue
        rejected = [o for o in others if o is not None]
        xc = feature_vector(chosen.scores)
        xr = [feature_vector(c.scores) for c in rejected]
        if xc is None or any(x is None for x in xr):
            continue
        hand = _hit_among([hand_reward(c) for c in (chosen, *rejected)])
        axes = {int(c.scores["axis_id"]) for c in (chosen, *rejected) if "axis_id" in c.scores}
        taps.append(
            Tap(
                day_id=day_id,
                panel_id=panel_id,
                tapped_at=group["created_at"],
                chosen_url=group["chosen"],
                rejected_urls=list(group["rejected"]),
                chosen=xc,
                rejected=[x for x in xr if x is not None],
                chosen_role=_role(chosen.scores),
                rejected_roles=[_role(c.scores) for c in rejected],
                chosen_emb=chosen.embedding,
                rejected_emb=[c.embedding for c in rejected],
                chosen_levels=knobs.levels_of(chosen.scores),
                rejected_levels=[knobs.levels_of(c.scores) for c in rejected],
                hand_correct=hand,
                axis=next(iter(axes)) if len(axes) == 1 else None,
                default_picked=_was_default(chosen.scores),
            )
        )
    return taps


@dataclass
class Rating:
    """The person's verdict on one image: 0 off, 1 ok, 2 great."""

    day_id: str
    panel_id: int
    url: str
    value: int
    rated_at: Any
    after_taps: int  # how many taps had been made when it was given (its place in time)
    scores_vector: np.ndarray
    emb: list[float] | None
    levels: np.ndarray | None


def _load_ratings(
    conn: psycopg.Connection[Any], user_id: str, storage: Storage | None, taps: list[Tap]
) -> list[Rating]:
    """The latest rating per image (a changed mind replaces the earlier one), oldest first."""
    rows = conn.execute(
        """
        SELECT DISTINCT ON (r.day_id, r.panel_id, r.url)
               r.day_id, r.panel_id, r.url, r.rating, r.created_at
        FROM image_ratings r JOIN days d ON d.job_id = r.day_id
        WHERE d.user_id = %s AND d.kind = 'daily'
        ORDER BY r.day_id, r.panel_id, r.url, r.created_at DESC, r.id DESC
        """,
        (user_id,),
    ).fetchall()
    tap_times = [t.tapped_at for t in taps]
    ratings: list[Rating] = []
    for row in sorted(rows, key=lambda r: r["created_at"]):
        candidate = _candidate(conn, storage, row["day_id"], row["url"])
        if candidate is None:
            continue
        vector = feature_vector(candidate.scores)
        if vector is None:
            continue
        ratings.append(
            Rating(
                day_id=row["day_id"],
                panel_id=row["panel_id"],
                url=row["url"],
                value=int(row["rating"]),
                rated_at=row["created_at"],
                after_taps=sum(1 for at in tap_times if at <= row["created_at"]),
                scores_vector=vector,
                emb=candidate.embedding,
                levels=knobs.levels_of(candidate.scores),
            )
        )
    return ratings


def _rating_rows(x: np.ndarray, value: int) -> list[np.ndarray]:
    """A rating as two cumulative-link observations sharing the utility `w . x`:
    P(rated at least ok) = sigmoid(w.x - b1) and P(rated great) = sigmoid(w.x - b2). Each is a
    row in the same logistic form the comparisons use, over `[w, b1, b2]` — a comparison says which
    of two images is better; a rating also says whether an image is good AT ALL, which a pick among
    three weak options never can."""
    rows = []
    for k, positive in enumerate((value >= 1, value >= 2)):
        row = np.concatenate([x, np.zeros(N_THRESHOLDS)])
        row[len(x) + k] = -1.0
        rows.append(row if positive else -row)
    return rows


def _scales(conn: psycopg.Connection[Any], user_id: str, taps: list[Tap]) -> np.ndarray:
    """Per-feature spread among a panel's own candidates, so a weight reads as
    "per typical difference between two candidates". Falls back to defaults
    where there isn't enough measured data yet."""
    rows = conn.execute(
        "SELECT c.day_id, c.panel_id, c.scores FROM candidates c "
        "JOIN days d ON d.job_id = c.day_id "
        "WHERE d.user_id = %s AND d.kind = 'daily' AND c.scores IS NOT NULL",
        (user_id,),
    ).fetchall()
    by_panel: dict[tuple[str, int], list[np.ndarray]] = {}
    for row in rows:
        v = feature_vector(dict(row["scores"]))
        if v is not None:
            by_panel.setdefault((row["day_id"], row["panel_id"]), []).append(v)
    centered = [np.array(vs) - np.mean(vs, axis=0) for vs in by_panel.values() if len(vs) >= 2]
    defaults = np.array([DEFAULT_SCALES[f] for f in FEATURES])
    n_scores = len(SCORE_FEATURES)
    if not centered:
        return defaults
    pooled = np.concatenate(centered)
    if pooled.shape[0] < 6:
        return defaults
    scales = defaults.copy()  # the embedding components are standardised: their scale stays 1
    std = pooled.std(axis=0)
    scales[:n_scores] = np.where(std > 1e-3, std, defaults[:n_scores])
    return scales


def should_activate(*, n_taps: int, edge: float) -> bool:
    """The held-out gate: enough taps AND a clear net edge over the default ranking."""
    return n_taps >= MIN_TAPS and edge >= MIN_EDGE


def _softmax_pick_prob(
    mu: np.ndarray, cov: np.ndarray, zc: np.ndarray, zr: list[np.ndarray]
) -> float:
    """Model's probability that the person picks `zc` over each of the others
    (Plackett-Luce winner probability), with each comparison's uncertainty shrinking it
    toward chance exactly as `model.predict_prob` does for a single pair."""
    total = 1.0
    for other in zr:
        diff = zc - other
        mean = float(mu @ diff)
        var = float(diff @ cov @ diff)
        total += float(np.exp(-np.clip(mean / np.sqrt(1.0 + np.pi * var / 8.0), -30.0, 30.0)))
    return 1.0 / total


def rebuild_preference(
    conn: psycopg.Connection[Any], user_id: str, storage: Storage | None = None
) -> dict[str, Any]:
    """Recompute the model + snapshots and write them. Returns a small summary."""
    taps = _load_taps(conn, user_id, storage)
    scales = _scales(conn, user_id, taps)
    n = len(taps)

    # The embedding components: the person's own candidates' embeddings, projected onto their
    # top principal directions (see `embedding.py`).
    seen_embeddings: dict[tuple[str, int, int], np.ndarray] = {}
    for i, tap in enumerate(taps):
        for j, emb in enumerate((tap.chosen_emb, *tap.rejected_emb)):
            if emb is not None:
                seen_embeddings[(tap.day_id, tap.panel_id, i * 10 + j)] = np.asarray(emb)
    projection = embedding.fit_projection(list(seen_embeddings.values()))

    def full(scores_vector: np.ndarray, emb: list[float] | None) -> np.ndarray:
        return np.concatenate([scores_vector, embedding.project(projection, emb)])

    n_f = len(FEATURES)
    n_k = 2 * knobs.K
    prior_var = np.concatenate([prior_variances(), np.full(N_THRESHOLDS, THRESHOLD_PRIOR_VAR)])
    knob_prior = np.concatenate(
        [np.full(n_k, model.PRIOR_VAR), np.full(N_THRESHOLDS, THRESHOLD_PRIOR_VAR)]
    )
    mu = np.zeros(n_f + N_THRESHOLDS)  # the last two are the rating cut points
    cov = np.diag(prior_var)
    feature_rows: list[np.ndarray] = []
    feature_tap: list[int] = []  # when each row happened, in taps (for the forgetting weights)
    mu_k = np.zeros(n_k + N_THRESHOLDS)
    cov_k = np.diag(knob_prior)
    knob_rows: list[np.ndarray] = []
    knob_tap: list[int] = []
    ratings = _load_ratings(conn, user_id, storage, taps)
    pending = list(ratings)  # oldest first; each is added when its turn in time comes
    knob_taps = 0
    knobs_weights = np.ones(0)
    others: dict[int, list[np.ndarray]] = {1: [], 2: []}  # feel-good: the other sampler's arm
    lean = np.zeros(knobs.K, dtype=int)
    seen = np.zeros((knobs.K, len(knobs.LEVELS)), dtype=bool)  # levels ever compared, per knob

    snapshots: list[dict[str, Any]] = []
    learned_hits: list[float] = []
    hand_hits: list[float] = []

    def refit(now: int, *, feature: bool, knob: bool) -> None:
        nonlocal mu, cov, mu_k, cov_k, knobs_weights
        if feature and feature_rows:
            ages = np.maximum(now - np.array(feature_tap), 0)
            mu, cov = model.fit(
                np.array(feature_rows),
                prior_var=prior_var,
                w0=mu,
                weights=model.decay_weights(ages, knobs.HALF_LIFE_TAPS),
            )
        if knob and knob_rows:
            knobs_weights = model.decay_weights(
                np.maximum(now - np.array(knob_tap), 0), knobs.HALF_LIFE_TAPS
            )
            mu_k, cov_k = model.fit(
                np.array(knob_rows), prior_var=knob_prior, w0=mu_k, weights=knobs_weights
            )

    def add_ratings_due(upto: int) -> None:
        """Add every rating given after at most `upto` taps that has not been added yet."""
        added_feature = added_knob = False
        while pending and pending[0].after_taps <= upto:
            rating = pending.pop(0)
            for row in _rating_rows(full(rating.scores_vector, rating.emb) / scales, rating.value):
                feature_rows.append(row)
                feature_tap.append(rating.after_taps)
                added_feature = True
            if rating.levels is not None:
                for row in _rating_rows(knobs.phi(rating.levels), rating.value):
                    knob_rows.append(row)
                    knob_tap.append(rating.after_taps)
                    added_knob = True
        if added_feature or added_knob:
            refit(upto, feature=added_feature, knob=added_knob)

    for t, tap in enumerate(taps):
        add_ratings_due(t)  # everything the person said before this tap
        zs = [
            full(tap.chosen, tap.chosen_emb) / scales,
            *[full(x, e) / scales for x, e in zip(tap.rejected, tap.rejected_emb, strict=True)],
        ]
        # Asked BEFORE learning from this tap: which would you pick?
        utilities = [float(mu[:n_f] @ z) for z in zs]
        correct = _hit_among(utilities)
        prob = _softmax_pick_prob(mu[:n_f], cov[:n_f, :n_f], zs[0], zs[1:])
        if t >= WARMUP:
            learned_hits.append(correct)
            hand_hits.append(tap.hand_correct)

        for other in zs[1:]:
            feature_rows.append(np.concatenate([zs[0] - other, np.zeros(N_THRESHOLDS)]))
            feature_tap.append(t)
        refit(t, feature=True, knob=False)

        if tap.chosen_levels is not None and all(lv is not None for lv in tap.rejected_levels):
            for other_levels in tap.rejected_levels:
                assert other_levels is not None
                knob_rows.append(
                    np.concatenate(
                        [
                            knobs.phi(tap.chosen_levels) - knobs.phi(other_levels),
                            np.zeros(N_THRESHOLDS),
                        ]
                    )
                )
                knob_tap.append(t)
            knob_taps += 1
            by_role = {
                role: levels
                for role, levels in zip(
                    (tap.chosen_role, *tap.rejected_roles),
                    (tap.chosen_levels, *tap.rejected_levels),
                    strict=True,
                )
                if role is not None and levels is not None
            }
            if 1 in by_role and 2 in by_role:  # a round the bandit drew both exploratory arms for
                others[1].append(knobs.phi(by_role[2]))
                others[2].append(knobs.phi(by_role[1]))
            for levels in (tap.chosen_levels, *tap.rejected_levels):
                assert levels is not None
                for axis_index, level in enumerate(levels):
                    seen[axis_index, int(level) + MAX_LEVEL] = True
            refit(t, feature=False, knob=True)
            lean = knobs.step_lean(lean, mu_k[:n_k], cov_k[:n_k, :n_k], knob_taps, seen)

        snapshots.append(
            {
                "tap": t + 1,
                "tap_row": tap,
                "prob": prob,
                "correct": correct,
                "mu": mu[:n_f].tolist(),
                "sd": np.sqrt(np.diag(cov)[:n_f]).tolist(),
                "delta": (zs[0] - np.mean(zs[1:], axis=0)).tolist(),
            }
        )

    add_ratings_due(n + 1)  # anything said after the last tap
    lean = (
        knobs.step_lean(lean, mu_k[:n_k], cov_k[:n_k, :n_k], knob_taps, seen) if knob_taps else lean
    )
    mu_w, cov_w = mu[:n_f], cov[:n_f, :n_f]
    mu_kw, cov_kw = mu_k[:n_k], cov_k[:n_k, :n_k]

    learned_acc = float(np.mean(learned_hits)) if learned_hits else None
    hand_acc = float(np.mean(hand_hits)) if hand_hits else None
    edge = float(sum(lh - hh for lh, hh in zip(learned_hits, hand_hits, strict=True)))
    learned_wins = sum(1 for lh, hh in zip(learned_hits, hand_hits, strict=True) if lh > hh)
    hand_wins = sum(1 for lh, hh in zip(learned_hits, hand_hits, strict=True) if lh < hh)
    active = should_activate(n_taps=n, edge=edge)

    sd = np.sqrt(np.diag(cov_w))
    z = {a: float(mu_w[FEATURES.index(a)] / sd[FEATURES.index(a)]) for a in AXES}
    lean_out = {axis: int(lean[i]) for i, axis in enumerate(AXES)}
    default_flags = [t.default_picked for t in taps if t.default_picked is not None]
    set_sizes = [1 + len(t.rejected) for t in taps]
    accuracy = {
        "learned": learned_acc,
        "hand": hand_acc,
        "n_eval": len(learned_hits),
        "min_taps": MIN_TAPS,
        "warmup": WARMUP,
        "edge": edge,
        "min_edge": MIN_EDGE,
        "learned_wins": learned_wins,
        "hand_wins": hand_wins,
        # How often the person's pick was the app's own first choice over ALL taps —
        # including the warm-up ones the comparison above skips.
        "overall_hand": float(np.mean([t.hand_correct for t in taps])) if taps else None,
        "overall_n": n,
        # KPI: how often the person picks the DEFAULT option (the one with no step either
        # side of their default) over the two variations. `chance` is 1 / options shown.
        "default_pick": float(np.mean(default_flags)) if default_flags else None,
        "default_pick_n": len(default_flags),
        "default_pick_chance": float(np.mean([1.0 / s for s in set_sizes])) if set_sizes else None,
        # What the person has said about single images (0 off, 1 ok, 2 great), and where the
        # model puts the cut points between them (in the utility's own units).
        "ratings": {
            "n": len(ratings),
            "off": sum(1 for x in ratings if x.value == 0),
            "ok": sum(1 for x in ratings if x.value == 1),
            "great": sum(1 for x in ratings if x.value == 2),
            "thresholds": [round(float(x), 4) for x in mu[n_f:]] if ratings else None,
        },
    }
    # The bandit's posterior: the same rows, but each counts for `ETA` of a comparison (the
    # tempering in Feel-Good Thompson Sampling), so its draws are wider than the plain posterior.
    if knob_rows:
        tempered_weights = knobs_weights * bandit.ETA
        mu_ts, cov_ts = model.fit(
            np.array(knob_rows), prior_var=knob_prior, weights=tempered_weights
        )
        ts_json: dict[str, object] | None = bandit.Posterior(
            mu=mu_ts[:n_k],
            cov=cov_ts[:n_k, :n_k],
            center_mu=mu_kw,
            other_arm={j: np.array(v) for j, v in others.items() if v},
        ).to_json()
    else:
        ts_json = None
    knob_state = {
        "ts": ts_json,
        "mu": mu_kw.tolist(),
        "cov": cov_kw.tolist(),
        "n_taps": knob_taps,
        "half_life": knobs.HALF_LIFE_TAPS,
        "enter_p": knobs.ENTER_P,
        "keep_p": knobs.KEEP_P,
        "min_taps": knobs.MIN_TAPS_LEAN,
        "uncertainty": knobs.uncertainty(cov_kw, lean),
        "guess": {
            a: int(v)
            for a, v in zip(
                AXES, knobs.guess_levels(lean, mu_kw, cov_kw, knob_taps, seen), strict=True
            )
        },
        "best": {a: knobs.best_level(mu_kw, cov_kw, i, seen) for i, a in enumerate(AXES)},
        "curves": {a: knobs.curve(mu_kw, cov_kw, i, seen) for i, a in enumerate(AXES)},
        "seen": {
            a: [lv for lv in knobs.LEVELS if seen[i, lv + MAX_LEVEL]] for i, a in enumerate(AXES)
        },
    }

    conn.execute(
        """
        INSERT INTO preference_models
            (user_id, features, mu, cov, scales, n_taps, active, lean, z, accuracy, knobs, embed,
             updated_at)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, now())
        ON CONFLICT (user_id) DO UPDATE SET
            features = EXCLUDED.features, mu = EXCLUDED.mu, cov = EXCLUDED.cov,
            scales = EXCLUDED.scales, n_taps = EXCLUDED.n_taps, active = EXCLUDED.active,
            lean = EXCLUDED.lean, z = EXCLUDED.z, accuracy = EXCLUDED.accuracy,
            knobs = EXCLUDED.knobs, embed = EXCLUDED.embed, updated_at = now()
        """,
        (
            user_id,
            Json(list(FEATURES)),
            Json(mu_w.tolist()),
            Json(cov_w.tolist()),
            Json(scales.tolist()),
            n,
            active,
            Json(lean_out),
            Json(z),
            Json(accuracy),
            Json(knob_state),
            Json(projection.to_json() if projection is not None else {}),
        ),
    )
    conn.execute("DELETE FROM preference_snapshots WHERE user_id = %s", (user_id,))
    for s in snapshots:
        tap = s["tap_row"]
        conn.execute(
            """
            INSERT INTO preference_snapshots
                (user_id, tap_index, day_id, panel_id, chosen_url, rejected_url, rejected_urls,
                 prob, correct, hand_correct, mu, sd, delta, axis, tapped_at, default_picked)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                user_id,
                s["tap"],
                tap.day_id,
                tap.panel_id,
                tap.chosen_url,
                tap.rejected_urls[0],
                Json(tap.rejected_urls),
                s["prob"],
                s["correct"],
                tap.hand_correct,
                Json(s["mu"]),
                Json(s["sd"]),
                Json(s["delta"]),
                tap.axis,
                tap.tapped_at,
                tap.default_picked,
            ),
        )
    return {"n_taps": n, "active": active, "accuracy": accuracy, "lean": lean_out}
