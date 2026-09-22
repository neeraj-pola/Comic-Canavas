"""The eval gate — the safety rail that decides whether a freshly-trained
artifact (reward model, script DPO, prompt-writer GRPO, diffusion-DPO)
actually ships, or stays behind the current champion. Pure decision
logic, deliberately with no model-specific knowledge of how any of the
numbers it's given were produced — callers gather the real metrics and
hand them to `decide()`.

Four rules, all must pass to promote:
1. reward accuracy >= champion + 0.01
2. script win-rate >= 0.55
3. critic mean not down (>= champion)
4. cost/strip not up more than 10%
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import psycopg
from dotenv import find_dotenv, load_dotenv

SCRIPT_WIN_RATE_THRESHOLD = 0.55
REWARD_ACCURACY_MARGIN = 0.01
MAX_COST_INCREASE_PCT = 10.0


@dataclass(frozen=True)
class GateInputs:
    reward_accuracy: float
    champion_reward_accuracy: float
    script_win_rate: float
    critic_mean: float
    champion_critic_mean: float
    cost_per_strip: float
    champion_cost_per_strip: float


@dataclass(frozen=True)
class GateDecision:
    promote: bool
    reasons: list[str] = field(default_factory=list)


def decide(inputs: GateInputs) -> GateDecision:
    reasons = []

    required_reward_accuracy = inputs.champion_reward_accuracy + REWARD_ACCURACY_MARGIN
    if inputs.reward_accuracy < required_reward_accuracy:
        reasons.append(
            f"reward accuracy {inputs.reward_accuracy:.4f} does not beat champion "
            f"{inputs.champion_reward_accuracy:.4f} by >= {REWARD_ACCURACY_MARGIN}"
        )

    if inputs.script_win_rate < SCRIPT_WIN_RATE_THRESHOLD:
        reasons.append(
            f"script win-rate {inputs.script_win_rate:.4f} is below the "
            f"{SCRIPT_WIN_RATE_THRESHOLD} threshold"
        )

    if inputs.critic_mean < inputs.champion_critic_mean:
        reasons.append(
            f"critic mean {inputs.critic_mean:.4f} is down from champion "
            f"{inputs.champion_critic_mean:.4f}"
        )

    if inputs.champion_cost_per_strip > 0:
        cost_increase_pct = (
            (inputs.cost_per_strip - inputs.champion_cost_per_strip)
            / inputs.champion_cost_per_strip
            * 100
        )
        if cost_increase_pct > MAX_COST_INCREASE_PCT:
            reasons.append(
                f"cost/strip up {cost_increase_pct:.1f}%, exceeds the "
                f"{MAX_COST_INCREASE_PCT}% limit"
            )

    return GateDecision(promote=not reasons, reasons=reasons)


def record_decision(training_run_id: int, decision: GateDecision) -> None:
    """Writes the real decision + reasons onto the real `training_runs` row."""
    load_dotenv(find_dotenv(usecwd=True))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError("DATABASE_URL not set (see .env.example)")
    with psycopg.connect(database_url) as conn:
        conn.execute(
            "UPDATE training_runs SET decision = %s, reasons = %s WHERE id = %s",
            ("promote" if decision.promote else "reject", decision.reasons, training_run_id),
        )
        conn.commit()


__all__ = ["GateDecision", "GateInputs", "decide", "record_decision"]
