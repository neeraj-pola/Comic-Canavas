"""Anti-hacking guards for the critic's reward score.

Identity naturally rewards large, clear faces, which a close framing
structurally provides more of than a wide one — left unchecked, this risks
the critic pushing every panel toward tight face crops across retries,
eroding the script's deliberate framing variety. `framing_diversity_bonus`
adds a small bonus for the `wide`/`medium` framings the script actually
asked for, offsetting that bias.

`has_detected_text` (via `rapidocr`) penalizes candidates with garbled
in-panel text, a diffusion-model artifact none of the other critic signals
would catch — dialogue lives in composed captions/bubbles, not the generated
image itself.

Capping any single signal's weight is handled in `reward_head.py`'s
`_clip01`, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    import numpy as np
    from rapidocr import RapidOCR

FRAMING_BONUS: dict[Literal["wide", "medium", "close"], float] = {
    "wide": 0.05,
    "medium": 0.02,
    "close": 0.0,
}

TEXT_PENALTY = 0.15

_engine: RapidOCR | None = None


def _get_engine() -> RapidOCR:
    global _engine
    if _engine is None:
        from rapidocr import RapidOCR as _RapidOCR

        _engine = _RapidOCR()
    return _engine


def has_detected_text(image_bgr: np.ndarray) -> bool:
    """`True` if RapidOCR finds any text region in the image. `result.boxes`
    is `None` when nothing is detected (not an empty list), handled
    defensively here."""
    result = _get_engine()(image_bgr)
    boxes = getattr(result, "boxes", None)
    return boxes is not None and len(boxes) > 0


def framing_diversity_bonus(framing: Literal["wide", "medium", "close"] | None) -> float:
    """`framing` is the panel's script-assigned framing (`None` when no
    script is wired) — returns `0.0` in that case rather than guessing."""
    if framing is None:
        return 0.0
    return FRAMING_BONUS.get(framing, 0.0)


def text_penalty(has_text: bool) -> float:
    return TEXT_PENALTY if has_text else 0.0


def guarded_reward(
    base_reward: float,
    *,
    framing: Literal["wide", "medium", "close"] | None = None,
    has_text: bool = False,
) -> float:
    """Applies the framing bonus and text penalty on top of the reward
    head's base score, clipped at `0.0` so a text penalty can't push a score
    negative."""
    return max(0.0, base_reward + framing_diversity_bonus(framing) - text_penalty(has_text))
