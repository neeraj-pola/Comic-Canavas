"""Candidates that differ on purpose.

Instead of three random seeds of the same prompt, each panel's three
candidates vary along ONE axis — warmer vs cooler, closer vs wider, subtler
vs more exaggerated expression — so a tap teaches a *direction* the person
likes. Which axis a panel explores is chosen by what the model is least sure
about. Once the model is confident about an axis, that lean is baked into
every candidate's prompt by default ("the prompt writer leans that way")
while the three still vary around it, so it keeps checking.
"""

from __future__ import annotations

from dataclasses import dataclass

from contracts import ImagePrompt

AXES = ("warmth", "closeness", "expression")
AXIS_IDS = {name: i for i, name in enumerate(AXES)}
MAX_LEVEL = 2

# level -> prompt phrase. Level 0 gets a NEUTRAL phrase on the knob a panel varies, so all
# three options carry a phrase about that knob — otherwise a bare "as usual" against two
# phrased options can look exactly like a taste for whatever the phrase describes, when the
# real effect is just "any phrase beats none."
PHRASES: dict[str, dict[int, str]] = {
    "warmth": {
        0: "a balanced, neutral color palette",
        -2: "a strongly cool color palette: cold blue and teal light",
        -1: "a cool color palette with blue-green tones",
        1: "a warm color palette with golden orange tones",
        2: "a strongly warm color palette: rich amber and orange light",
    },
    "closeness": {
        0: "a standard medium-distance view",
        -2: "the camera pulled far back, the character small in a large environment",
        -1: "a slightly wider view than usual, more of the environment visible",
        1: "a slightly closer view than usual, the character larger in the frame",
        2: "a tight close-up view, the character filling much of the frame",
    },
    "expression": {
        0: "a natural, everyday expression",
        -2: "a very subtle, restrained, almost neutral expression",
        -1: "a subdued, understated expression",
        1: "a strongly expressive, animated expression",
        2: "an extremely exaggerated, dramatic comic expression",
    },
}

# What each direction is called in the UI (positive level, negative level).
DIRECTION_LABELS: dict[str, tuple[str, str]] = {
    "warmth": ("warmer", "cooler"),
    "closeness": ("closer", "wider"),
    "expression": ("bolder", "subtler"),
}


@dataclass(frozen=True)
class AxisPlan:
    axis: str  # the axis this panel's candidates vary along
    lean: dict[str, int]  # per-axis default lean (-1, 0, +1) from what's been learned


def _clamp(level: int) -> int:
    return max(-MAX_LEVEL, min(MAX_LEVEL, level))


def plan_for_panel(
    panel_id: int,
    *,
    z: dict[str, float] | None = None,
    lean: dict[str, int] | None = None,
    n_taps: int = 0,
    uncertainty: dict[str, float] | None = None,
) -> AxisPlan:
    """Rank axes by how unsure the model is once it has a few taps — by the knob model's
    `uncertainty` (largest first) when given, else by the smallest |z|; before that, a
    fixed order. Panel k explores the k-th axis, so a
    day's four panels spread across all three axes instead of repeating one."""
    order = list(AXES)
    if uncertainty and n_taps >= 5:
        order.sort(key=lambda axis: -uncertainty.get(axis, 0.0))  # least sure first
    elif z and n_taps >= 5:
        order.sort(key=lambda axis: abs(z.get(axis, 0.0)))
    return AxisPlan(
        axis=order[(panel_id - 1) % len(order)], lean={a: (lean or {}).get(a, 0) for a in AXES}
    )


def levels_for_candidate(plan: AxisPlan, k: int, n: int = 3) -> dict[str, int]:
    """Level applied on every axis for candidate `k` of `n`: the lean, plus a
    -1/0/+1 offset on the panel's own axis."""
    offset = k - (n - 1) // 2
    levels = dict(plan.lean)
    levels[plan.axis] = _clamp(plan.lean.get(plan.axis, 0) + offset)
    return levels


def variant_prompt(
    prompt: ImagePrompt, plan: AxisPlan, k: int, n: int = 3, *, same_seed: bool = False
) -> tuple[ImagePrompt, dict[str, float]]:
    """The k-th candidate's prompt (base prompt + its axis phrases) and the
    preset scores recording what was asked for. With `same_seed` all options share the
    prompt's seed, so they differ only in the phrase (the person compares what the phrase
    changed, not different random draws); otherwise seeds are `base + k`. Candidate keys
    include a tag of the prompt text, so shared seeds never collide."""
    levels = levels_for_candidate(plan, k, n)
    # A knob at level 0 is spoken to only when it is the one being varied (see PHRASES).
    clauses = [
        PHRASES[axis][level] for axis, level in levels.items() if level != 0 or axis == plan.axis
    ]
    positive = prompt.positive
    if clauses:
        positive = f"{positive.rstrip('. ')}. Also: {'; '.join(clauses)}."
    seed = (prompt.seed or 0) + (0 if same_seed else k)
    variant = prompt.model_copy(update={"positive": positive, "seed": seed})
    preset = {
        "axis_id": float(AXIS_IDS[plan.axis]),
        "axis_level": float(levels[plan.axis]),
        "expression": float(levels["expression"]),
        # Every knob's intended level, so the knob model can learn from any candidate; and
        # which of the three this is (0 = the default, -1/+1 = a step either side), so a tap
        # on the default can be told apart from a tap on a variation.
        **{f"level_{axis}": float(level) for axis, level in levels.items()},
        "offset": float(k - (n - 1) // 2),
        # Marks candidates whose every option was phrased (see PHRASES); the knob model only
        # learns from these — earlier ones compared a plain prompt against phrased ones.
        "phrased": 1.0,
    }
    return variant, preset


def option_prompts(
    prompt: ImagePrompt,
    level_sets: list[dict[str, int]],
    roles: list[int],
    *,
    same_seed: bool = False,
) -> list[tuple[ImagePrompt, dict[str, float]]]:
    """The prompts (and preset scores) for options that were chosen as whole settings — the
    bandit's draws, which may differ on several knobs at once — instead of one knob stepped
    either way. A knob is spoken to (even at level 0, with its neutral phrase) exactly when the
    options disagree on it, so every phrase difference between the options is a real one.
    `roles`: 0 = the model's best guess, 1 and 2 = its two exploratory draws."""
    spoken = {a for a in AXES if len({levels[a] for levels in level_sets}) > 1}
    out: list[tuple[ImagePrompt, dict[str, float]]] = []
    for k, (levels, role) in enumerate(zip(level_sets, roles, strict=True)):
        clauses = [
            PHRASES[axis][level] for axis, level in levels.items() if level != 0 or axis in spoken
        ]
        positive = prompt.positive
        if clauses:
            positive = f"{positive.rstrip('. ')}. Also: {'; '.join(clauses)}."
        seed = (prompt.seed or 0) + (0 if same_seed else k)
        variant = prompt.model_copy(update={"positive": positive, "seed": seed})
        preset = {
            "expression": float(levels["expression"]),
            **{f"level_{axis}": float(level) for axis, level in levels.items()},
            "role": float(role),
            # 0 marks the best-guess option (what a "picked the default" tap is counted against).
            "offset": 0.0 if role == 0 else (1.0 if role == 1 else -1.0),
            "phrased": 1.0,
        }
        out.append((variant, preset))
    return out
