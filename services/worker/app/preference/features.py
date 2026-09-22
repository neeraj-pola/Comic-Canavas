"""The ten numbers the preference model can see about a candidate image.

Four are the existing critic signals (identity, style, alignment, detail),
already stored in `Candidate.scores`. Five are measured directly from the
pixels here (warmth, brightness, saturation, contrast, closeness). The last,
`expression`, can't be measured from pixels without another model, so it is
the *intended* expression level the candidate's prompt asked for (see
`axes.py`) — honest about being a construction, not a measurement.
"""

from __future__ import annotations

import cv2
import numpy as np

CRITIC_FEATURES = ("identity", "style", "alignment", "detail")
PIXEL_FEATURES = ("warmth", "brightness", "saturation", "contrast", "closeness")
AXIS_FEATURE = "expression"
# The vision judge's ratings (`critic/judge.py`): scene, face, body, outfit, overall.
JUDGE_FEATURES = tuple(f"judge_{d}" for d in ("scene", "face", "body", "outfit", "overall"))
# Everything a candidate's `scores` can give the model.
SCORE_FEATURES: tuple[str, ...] = (
    *CRITIC_FEATURES,
    *PIXEL_FEATURES,
    AXIS_FEATURE,
    *JUDGE_FEATURES,
)
# The last six are principal components of the image embedding (`embedding.py`) — not something
# a person can name, so the Learning page does not list them.
EMBED_DIMS = 6
EMBED_FEATURES = tuple(f"emb_{i}" for i in range(EMBED_DIMS))
FEATURES: tuple[str, ...] = (*SCORE_FEATURES, *EMBED_FEATURES)
JUDGE_NEUTRAL = 0.5  # a candidate with no judge ratings reads as "average" on every one

# Only used until a user has enough of their own candidates to measure the
# spread from; typical within-panel standard deviations of each feature.
DEFAULT_SCALES: dict[str, float] = {
    "identity": 0.15,
    "style": 0.10,
    "alignment": 0.05,
    "detail": 0.05,
    "warmth": 0.05,
    "brightness": 0.08,
    "saturation": 0.08,
    "contrast": 0.05,
    "closeness": 0.05,
    "expression": 1.0,
    **{name: 0.25 for name in JUDGE_FEATURES},
    **{name: 1.0 for name in EMBED_FEATURES},  # components are already standardised
}

# How far from zero each weight may plausibly be. The embedding components are many and
# opaque, so they get a tighter prior: they earn a weight only when the picks insist.
PRIOR_VARIANCE_DEFAULT = 1.0
PRIOR_VARIANCE_EMBED = 0.25


def prior_variances() -> np.ndarray:
    return np.array(
        [
            PRIOR_VARIANCE_EMBED if name in EMBED_FEATURES else PRIOR_VARIANCE_DEFAULT
            for name in FEATURES
        ]
    )


def pixel_features(
    image_bgr: np.ndarray, face_box: tuple[int, int, int, int] | None
) -> dict[str, float]:
    """Cheap, deterministic, CPU-only. `closeness` is the square root of the
    face's share of the frame (so it grows roughly linearly with how close
    the framing is), 0.0 when no face was found."""
    blue = float(image_bgr[..., 0].mean())
    red = float(image_bgr[..., 2].mean())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    closeness = 0.0
    if face_box is not None:
        x0, y0, x1, y1 = face_box
        height, width = image_bgr.shape[:2]
        closeness = float(np.sqrt(max(0, (x1 - x0) * (y1 - y0)) / (height * width)))

    return {
        "warmth": (red - blue) / 255.0,
        "brightness": float(gray.mean()) / 255.0,
        "saturation": float(hsv[..., 1].mean()) / 255.0,
        "contrast": float(gray.std()) / 255.0,
        "closeness": closeness,
    }


def feature_vector(scores: dict[str, float]) -> np.ndarray | None:
    """The `SCORE_FEATURES` of one candidate. `None` when any measured feature is missing (a
    candidate scored before this feature set existed) — callers fall back rather than guess.
    The judge's ratings may be absent (older candidates, or the call failed): those read as
    `JUDGE_NEUTRAL`, so every option of a panel that was never judged ties on them."""
    if any(name not in scores for name in (*CRITIC_FEATURES, *PIXEL_FEATURES)):
        return None
    return np.array(
        [
            float(scores.get(name, JUDGE_NEUTRAL if name in JUDGE_FEATURES else 0.0))
            for name in SCORE_FEATURES
        ],
        dtype=np.float64,
    )
