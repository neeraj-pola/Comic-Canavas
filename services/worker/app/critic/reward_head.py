"""Reward head — pipeline node 7's final combined score.

Combines the four critic signals (identity, style, alignment, detail) into
one number `nodes/critic.py` uses to pick the best of several candidates per
panel, as a hand-set linear combination — a separately-trained Bradley-Terry
model can replace these weights once real pick data exists.

Each input score is clipped to `[0, 1]` before weighting, so no single
upstream signal can exceed its own weight's maximum contribution by
returning a value outside its expected range.
"""

from __future__ import annotations

from contracts import Candidate

# Hand-set weights, sum to 1.0.
IDENTITY_WEIGHT = 0.35
STYLE_WEIGHT = 0.25
ALIGNMENT_WEIGHT = 0.25
DETAIL_WEIGHT = 0.15

WEIGHTS: dict[str, float] = {
    "identity": IDENTITY_WEIGHT,
    "style": STYLE_WEIGHT,
    "alignment": ALIGNMENT_WEIGHT,
    "detail": DETAIL_WEIGHT,
}


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


# The vision judge's ratings (`critic/judge.py`), 0-1 each. Kept here as plain names so this
# module stays free of any model or storage import.
JUDGE_KEYS = ("judge_scene", "judge_face", "judge_body", "judge_outfit", "judge_overall")
JUDGE_SHARE = 0.5


def score(candidate: Candidate) -> float:
    """Deterministic — the same `candidate.scores` always produces the same
    result. Missing keys default to `0.0`, so a signal that was never
    computed counts as a full penalty for its weight's share, not a free
    pass."""
    base = sum(
        weight * _clip01(candidate.scores.get(signal, 0.0)) for signal, weight in WEIGHTS.items()
    )
    judged = [_clip01(candidate.scores[key]) for key in JUDGE_KEYS if key in candidate.scores]
    if not judged:
        return base  # no judge score for this candidate: the four signals alone
    # Half the four signals, half the judge — chosen a priori (equal halves), not tuned.
    return (1.0 - JUDGE_SHARE) * base + JUDGE_SHARE * (sum(judged) / len(judged))
