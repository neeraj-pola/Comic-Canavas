"""`ml/evals/critic_identity.py`'s pure-math layer, `auc_score` — the
real accept-line metric, network-free and covered by `make test`.
`run_eval`/`main` need real assets + a real `LLM_JUDGE` provider and are
not exercised here (same split as `ml/evals/extractor.py`).
"""

from __future__ import annotations

import pytest

from ml.evals.critic_identity import auc_score


def test_auc_is_one_for_perfect_separation() -> None:
    assert auc_score([0.9, 0.8, 0.95], [0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_auc_is_zero_for_perfectly_inverted_separation() -> None:
    assert auc_score([0.1, 0.2, 0.3], [0.9, 0.8, 0.95]) == pytest.approx(0.0)


def test_auc_is_half_for_identical_distributions() -> None:
    assert auc_score([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) == pytest.approx(0.5)


def test_auc_handles_partial_overlap() -> None:
    # 2 of 3 positives beat all negatives, 1 ties with a negative.
    positive = [0.9, 0.8, 0.5]
    negative = [0.5, 0.3, 0.2]
    auc = auc_score(positive, negative)
    assert 0.5 < auc < 1.0
