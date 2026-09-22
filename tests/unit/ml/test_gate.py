"""ml/evals/gate.py's decide() — unit tests for each rule plus "a
synthetic regression is rejected". `record_decision`'s real DB write is
exercised separately (see test_gate_db.py for the real-Postgres round
trip) — kept separate so this file needs no database at all.
"""

from __future__ import annotations

from ml.evals.gate import GateInputs, decide

CHAMPION = GateInputs(
    reward_accuracy=0.80,
    champion_reward_accuracy=0.80,
    script_win_rate=0.60,
    critic_mean=0.70,
    champion_critic_mean=0.70,
    cost_per_strip=0.40,
    champion_cost_per_strip=0.40,
)


def _challenger(**overrides: float) -> GateInputs:
    return GateInputs(
        reward_accuracy=overrides.get("reward_accuracy", 0.82),
        champion_reward_accuracy=CHAMPION.champion_reward_accuracy,
        script_win_rate=overrides.get("script_win_rate", 0.60),
        critic_mean=overrides.get("critic_mean", 0.70),
        champion_critic_mean=CHAMPION.champion_critic_mean,
        cost_per_strip=overrides.get("cost_per_strip", 0.40),
        champion_cost_per_strip=CHAMPION.champion_cost_per_strip,
    )


def test_a_real_improvement_on_every_axis_is_promoted() -> None:
    decision = decide(_challenger(reward_accuracy=0.82))
    assert decision.promote is True
    assert decision.reasons == []


def test_reward_accuracy_must_beat_champion_by_at_least_the_margin() -> None:
    # Only 0.005 better — below the 0.01 margin.
    decision = decide(_challenger(reward_accuracy=0.805))
    assert decision.promote is False
    assert any("reward accuracy" in r for r in decision.reasons)


def test_reward_accuracy_exactly_at_the_margin_passes() -> None:
    decision = decide(_challenger(reward_accuracy=0.81))
    assert decision.promote is True


def test_script_win_rate_below_threshold_is_rejected() -> None:
    decision = decide(_challenger(script_win_rate=0.54))
    assert decision.promote is False
    assert any("win-rate" in r for r in decision.reasons)


def test_script_win_rate_at_threshold_passes() -> None:
    decision = decide(_challenger(script_win_rate=0.55))
    assert decision.promote is True


def test_critic_mean_regression_is_rejected() -> None:
    decision = decide(_challenger(critic_mean=0.65))
    assert decision.promote is False
    assert any("critic mean" in r for r in decision.reasons)


def test_cost_increase_over_10_percent_is_rejected() -> None:
    decision = decide(_challenger(cost_per_strip=0.45))  # +12.5%
    assert decision.promote is False
    assert any("cost/strip" in r for r in decision.reasons)


def test_cost_increase_at_exactly_10_percent_passes() -> None:
    decision = decide(_challenger(cost_per_strip=0.44))  # +10.0%
    assert decision.promote is True


def test_a_synthetic_regression_failing_every_rule_is_rejected_with_all_reasons() -> None:
    """The task's own literal accept line."""
    regression = GateInputs(
        reward_accuracy=0.50,
        champion_reward_accuracy=0.80,
        script_win_rate=0.30,
        critic_mean=0.40,
        champion_critic_mean=0.70,
        cost_per_strip=1.00,
        champion_cost_per_strip=0.40,
    )

    decision = decide(regression)

    assert decision.promote is False
    assert len(decision.reasons) == 4


def test_this_projects_own_real_reward_model_result_would_be_rejected() -> None:
    """A real reward-model result with learned_pairwise_accuracy tied
    with hand_set accuracy — no improvement at all, let alone by the
    required margin — must be rejected by the gate."""
    decision = decide(
        GateInputs(
            reward_accuracy=0.4286,
            champion_reward_accuracy=0.4286,
            script_win_rate=1.0,
            critic_mean=0.70,
            champion_critic_mean=0.70,
            cost_per_strip=0.40,
            champion_cost_per_strip=0.40,
        )
    )
    assert decision.promote is False
