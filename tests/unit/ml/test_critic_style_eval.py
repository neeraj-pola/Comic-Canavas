"""`ml/evals/critic_style.py`'s pure-math layer, `best_threshold_accuracy`
— network-free, covered by `make test`. `run_eval`/`main` need real
assets and are not exercised here (same split as
`ml/evals/critic_identity.py`).
"""

from __future__ import annotations

import pytest

from ml.evals.critic_style import best_threshold_accuracy


def test_perfect_separation_gives_accuracy_one() -> None:
    accuracy, _threshold = best_threshold_accuracy([0.9, 0.8, 0.95], [0.1, 0.2, 0.3])
    assert accuracy == pytest.approx(1.0)


def test_identical_distributions_cap_accuracy_at_half() -> None:
    accuracy, _threshold = best_threshold_accuracy([0.5, 0.5], [0.5, 0.5])
    assert accuracy <= 0.5


def test_threshold_is_within_the_observed_score_range() -> None:
    positive = [0.9, 0.8]
    negative = [0.2, 0.1]
    _accuracy, threshold = best_threshold_accuracy(positive, negative)
    assert min(positive + negative) <= threshold <= max(positive + negative)
